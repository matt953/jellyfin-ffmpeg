/*
 * JM reference H.264/MVC decoder bridge implementation
 * This file includes ldecod headers (which conflict with FFmpeg headers).
 * It implements the bridge API defined in libldecod_jm_bridge.h.
 *
 * Uses the RawPicOutput callback to bypass ldecod's img2buf conversion,
 * reading directly from internal imgpel** arrays with NEON-optimized
 * uint16->uint8 narrowing.
 */

#include <ldecod/h264decoder.h>
#include <ldecod/h264decoder_mt.h>
#include <ldecod/configfile.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "libldecod_jm_bridge.h"

#define RING_BUF_SIZE (4 * 1024 * 1024)

/* ---- Per-view FIFO queue for MVC view pairing ---- */

typedef struct ViewFrameNode {
    JMDecodedFrame frame;
    int poc;
    struct ViewFrameNode *next;
} ViewFrameNode;

/* ---- Linked-list frame queue (never blocks on enqueue) ---- */

typedef struct FrameNode {
    JMDecodedFrame frame;
    struct FrameNode *next;
} FrameNode;

struct JMDecoderContext {
    pthread_t decoder_thread;
    int thread_started;

    InputParameters inp;
    ANNEXB_t *annex_b;
    MVCDecoderMT *mt;       /* MT decoder context (NULL = single-threaded) */

    /* Linked-list frame queue: enqueue never blocks, dequeue may wait */
    FrameNode *fq_head;
    FrameNode *fq_tail;
    int fq_count;
    pthread_mutex_t fq_mutex;
    pthread_cond_t fq_cond;

    /* FIFO view pairing: each view pushes to its own queue.
     * When both queues are non-empty, pop one from each, combine SBS, enqueue.
     * This avoids POC-matching which breaks across GOP boundaries. */
    ViewFrameNode *view_head[2];  /* Per-view FIFO heads */
    ViewFrameNode *view_tail[2];  /* Per-view FIFO tails */
    pthread_mutex_t pair_mutex;

    volatile int finished;
    volatile int error;
};

/* ---- NEON-optimized uint16 -> uint8 saturating narrow ---- */

/* Narrow one row of imgpel (uint16) to uint8 with saturation (clipping to 0-255) */
static inline void narrow_row(const imgpel *src, uint8_t *dst, int width)
{
#ifdef __aarch64__
    int x = 0;
    for (; x + 8 <= width; x += 8) {
        uint16x8_t v = vld1q_u16((const uint16_t *)(src + x));
        vst1_u8(dst + x, vqmovn_u16(v));
    }
    for (; x < width; x++)
        dst[x] = src[x] > 255 ? 255 : (uint8_t)src[x];
#else
    int x;
    for (x = 0; x < width; x++)
        dst[x] = src[x] > 255 ? 255 : (uint8_t)src[x];
#endif
}

/* Narrow a cropped plane from imgpel** rows to packed uint8 buffer */
static void narrow_plane(imgpel **rows, int crop_x, int crop_y,
                         uint8_t *dst, int dst_stride, int width, int height)
{
    int y;
    for (y = 0; y < height; y++)
        narrow_row(rows[crop_y + y] + crop_x, dst + y * dst_stride, width);
}

/* Narrow a single view from RawDecodedPic into a JMDecodedFrame */
static int narrow_view_to_frame(const RawDecodedPic *pic, JMDecodedFrame *frame)
{
    int w   = pic->width;
    int h   = pic->height;
    int wcr = pic->width_cr;
    int hcr = pic->height_cr;

    memset(frame, 0, sizeof(*frame));
    frame->width  = w;
    frame->height = h;
    frame->poc    = pic->poc;
    frame->strides[0] = w;
    frame->strides[1] = wcr;
    frame->strides[2] = wcr;

    frame->planes[0] = (uint8_t *)malloc(w * h);
    if (!frame->planes[0]) return -1;
    narrow_plane(pic->imgY, pic->crop_x, pic->crop_y,
                 frame->planes[0], w, w, h);

    if (wcr > 0) {
        frame->planes[1] = (uint8_t *)malloc(wcr * hcr);
        frame->planes[2] = (uint8_t *)malloc(wcr * hcr);
        if (frame->planes[1])
            narrow_plane(pic->imgU, pic->crop_x_cr, pic->crop_y_cr,
                         frame->planes[1], wcr, wcr, hcr);
        if (frame->planes[2])
            narrow_plane(pic->imgV, pic->crop_x_cr, pic->crop_y_cr,
                         frame->planes[2], wcr, wcr, hcr);
    }
    return 0;
}

/* ---- Frame queue (linked list, never blocks on enqueue) ---- */

static void enqueue_frame(JMDecoderContext *ctx, JMDecodedFrame *frame)
{
    FrameNode *node = (FrameNode *)malloc(sizeof(FrameNode));
    if (!node) return;
    node->frame = *frame;
    node->next = NULL;

    pthread_mutex_lock(&ctx->fq_mutex);
    if (ctx->fq_tail)
        ctx->fq_tail->next = node;
    else
        ctx->fq_head = node;
    ctx->fq_tail = node;
    ctx->fq_count++;
    pthread_cond_signal(&ctx->fq_cond);
    pthread_mutex_unlock(&ctx->fq_mutex);
}

/* ---- FIFO view pairing helpers ---- */

/* Push a frame into a per-view FIFO queue */
static void view_fifo_push(JMDecoderContext *ctx, int view_id, JMDecodedFrame *frame, int poc)
{
    ViewFrameNode *node = (ViewFrameNode *)malloc(sizeof(ViewFrameNode));
    if (!node) return;
    node->frame = *frame;
    node->poc = poc;
    node->next = NULL;
    if (ctx->view_tail[view_id])
        ctx->view_tail[view_id]->next = node;
    else
        ctx->view_head[view_id] = node;
    ctx->view_tail[view_id] = node;
}

/* Forward declaration */
static void combine_and_enqueue(JMDecoderContext *ctx,
                                JMDecodedFrame *view0, JMDecodedFrame *view1);

/* Try to pair: if both view FIFOs are non-empty, combine and enqueue */
static void try_pair_views(JMDecoderContext *ctx)
{
    while (ctx->view_head[0] && ctx->view_head[1]) {
        ViewFrameNode *n0 = ctx->view_head[0];
        ViewFrameNode *n1 = ctx->view_head[1];

        if (n0->poc != n1->poc)
            fprintf(stderr, "[PAIR] WARNING: view0 poc=%d != view1 poc=%d (FIFO order mismatch)\n",
                    n0->poc, n1->poc);

        combine_and_enqueue(ctx, &n0->frame, &n1->frame);

        /* Pop both */
        ctx->view_head[0] = n0->next;
        if (!ctx->view_head[0]) ctx->view_tail[0] = NULL;
        jm_frame_free(&n0->frame);
        free(n0);

        ctx->view_head[1] = n1->next;
        if (!ctx->view_head[1]) ctx->view_tail[1] = NULL;
        jm_frame_free(&n1->frame);
        free(n1);
    }
}

/* Combine two single-view frames into one SBS frame and enqueue */
static void combine_and_enqueue(JMDecoderContext *ctx,
                                JMDecodedFrame *view0, JMDecodedFrame *view1)
{
    JMDecodedFrame sbs;
    int w   = view0->width;
    int h   = view0->height;
    int hw  = w / 2;  /* chroma width per view */
    int hh  = h / 2;  /* chroma height */
    int y;

    memset(&sbs, 0, sizeof(sbs));
    sbs.width  = w;
    sbs.height = h;
    sbs.is_mvc = 1;
    sbs.poc    = view0->poc;
    sbs.strides[0] = w * 2;
    sbs.strides[1] = hw * 2;
    sbs.strides[2] = hw * 2;

    /* Y plane */
    sbs.planes[0] = (uint8_t *)malloc(w * 2 * h);
    if (!sbs.planes[0]) return;
    for (y = 0; y < h; y++) {
        memcpy(sbs.planes[0] + y * w * 2,     view0->planes[0] + y * w, w);
        memcpy(sbs.planes[0] + y * w * 2 + w, view1->planes[0] + y * w, w);
    }

    /* U plane */
    if (view0->planes[1] && view1->planes[1]) {
        sbs.planes[1] = (uint8_t *)malloc(hw * 2 * hh);
        if (sbs.planes[1]) {
            for (y = 0; y < hh; y++) {
                memcpy(sbs.planes[1] + y * hw * 2,      view0->planes[1] + y * hw, hw);
                memcpy(sbs.planes[1] + y * hw * 2 + hw,  view1->planes[1] + y * hw, hw);
            }
        }
    }

    /* V plane */
    if (view0->planes[2] && view1->planes[2]) {
        sbs.planes[2] = (uint8_t *)malloc(hw * 2 * hh);
        if (sbs.planes[2]) {
            for (y = 0; y < hh; y++) {
                memcpy(sbs.planes[2] + y * hw * 2,      view0->planes[2] + y * hw, hw);
                memcpy(sbs.planes[2] + y * hw * 2 + hw,  view1->planes[2] + y * hw, hw);
            }
        }
    }

    enqueue_frame(ctx, &sbs);
}

/* ---- Raw picture callback (called from ldecod's write_out_picture) ---- */

static void raw_pic_handler(void *user_data, const RawDecodedPic *pic)
{
    JMDecoderContext *ctx = (JMDecoderContext *)user_data;
    int view_id = pic->view_id >= 0 ? (pic->view_id & 0xffff) : -1;
    int poc = pic->poc;

    fprintf(stderr, "[RAW] view_id=%d poc=%d\n", view_id, poc);

    pthread_mutex_lock(&ctx->pair_mutex);

    if (view_id == 0 || view_id == 1) {
        /* Narrow this view's picture and push to its FIFO */
        JMDecodedFrame frame;
        if (narrow_view_to_frame(pic, &frame) < 0) goto done;
        view_fifo_push(ctx, view_id, &frame, poc);

        /* Try to pair: if both FIFOs have entries, combine oldest from each */
        try_pair_views(ctx);
    } else {
        /* Non-MVC frame (view_id < 0) — output as single view */
        JMDecodedFrame frame;
        if (narrow_view_to_frame(pic, &frame) < 0) goto done;
        frame.is_mvc = 0;
        enqueue_frame(ctx, &frame);
    }

done:
    pthread_mutex_unlock(&ctx->pair_mutex);
}

/* ---- Public API ---- */

JMDecoderContext *jm_decoder_open(int decode_all_layers)
{
    int iRet;
    JMDecoderContext *ctx = (JMDecoderContext *)calloc(1, sizeof(JMDecoderContext));
    if (!ctx)
        return NULL;

    memset(&ctx->inp, 0, sizeof(InputParameters));
    strcpy(ctx->inp.infile, "");
    strcpy(ctx->inp.outfile, "");
    strcpy(ctx->inp.reffile, "");
    ctx->inp.FileFormat = 0;

    /* Set non-zero defaults from Map[] in configfile.h.
     * The standalone decoder sets these via InitParams(Map) called from
     * ParseCommand. We must replicate them here since we skip ParseCommand. */
    ctx->inp.poc_scale = 2;
    ctx->inp.ref_poc_gap = 2;
    ctx->inp.poc_gap = 2;
    ctx->inp.write_uv = 1;
    ctx->inp.intra_profile_deblocking = 1;
    ctx->inp.dpb_plus[0] = 1;
    ctx->inp.silent = 1;
    ctx->inp.bDisplayDecParams = 1;

#if (MVC_EXTENSION_ENABLE)
    ctx->inp.DecodeAllLayers = decode_all_layers;
#endif

    pthread_mutex_init(&ctx->fq_mutex, NULL);
    pthread_cond_init(&ctx->fq_cond, NULL);
    pthread_mutex_init(&ctx->pair_mutex, NULL);

    /* Use MT dual-instance decoder: two decoder threads + NALU splitter */
    fprintf(stderr, "[BRIDGE] About to call OpenDecoderMT\n");
    ctx->mt = OpenDecoderMT(&ctx->inp, RING_BUF_SIZE, &ctx->annex_b);
    fprintf(stderr, "[BRIDGE] OpenDecoderMT returned %p\n", (void *)ctx->mt);
    if (!ctx->mt) {
        pthread_mutex_destroy(&ctx->fq_mutex);
        pthread_cond_destroy(&ctx->fq_cond);
        pthread_mutex_destroy(&ctx->pair_mutex);
        free(ctx);
        return NULL;
    }

    /* Set raw picture callback — called from both view threads */
    SetRawPicOutputMT(ctx->mt, raw_pic_handler, ctx);

    /* Start threads AFTER callbacks are set */
    if (StartDecoderMT(ctx->mt) != 0) {
        CloseDecoderMT(ctx->mt);
        ctx->mt = NULL;
        pthread_mutex_destroy(&ctx->fq_mutex);
        pthread_cond_destroy(&ctx->fq_cond);
        pthread_mutex_destroy(&ctx->pair_mutex);
        free(ctx);
        return NULL;
    }

    return ctx;
}

int jm_decoder_feed(JMDecoderContext *ctx, const uint8_t *data, int size)
{
    if (!ctx || !ctx->annex_b || size <= 0)
        return 0;

    /* Use try_feed in a loop, checking for decoder completion */
    int offset = 0;
    while (offset < size) {
        int ret = annex_b_ring_try_feed(ctx->annex_b,
                                         (const byte *)data + offset,
                                         size - offset);
        if (ret > 0) {
            offset += ret;
        } else {
            /* Ring full — check if decoders finished */
            if (ctx->mt && ctx->mt->view_finished[0] && ctx->mt->view_finished[1])
                return -1;
            usleep(1000);
        }
    }
    return offset;
}

int jm_decoder_try_feed(JMDecoderContext *ctx, const uint8_t *data, int size)
{
    if (!ctx || !ctx->annex_b || size <= 0)
        return 0;
    return annex_b_ring_try_feed(ctx->annex_b, (const byte *)data, size);
}

int jm_decoder_get_frame(JMDecoderContext *ctx, JMDecodedFrame *frame)
{
    FrameNode *node;
    if (!ctx)
        return 0;
    pthread_mutex_lock(&ctx->fq_mutex);
    node = ctx->fq_head;
    if (!node) {
        pthread_mutex_unlock(&ctx->fq_mutex);
        return 0;
    }
    *frame = node->frame;
    ctx->fq_head = node->next;
    if (!ctx->fq_head)
        ctx->fq_tail = NULL;
    ctx->fq_count--;
    pthread_mutex_unlock(&ctx->fq_mutex);
    free(node);
    return 1;
}

int jm_decoder_get_frame_blocking(JMDecoderContext *ctx, JMDecodedFrame *frame)
{
    FrameNode *node;
    if (!ctx)
        return 0;
    pthread_mutex_lock(&ctx->fq_mutex);
    while (!ctx->fq_head) {
        if (ctx->finished) {
            pthread_mutex_unlock(&ctx->fq_mutex);
            return 0;
        }
        pthread_cond_wait(&ctx->fq_cond, &ctx->fq_mutex);
    }
    node = ctx->fq_head;
    *frame = node->frame;
    ctx->fq_head = node->next;
    if (!ctx->fq_head)
        ctx->fq_tail = NULL;
    ctx->fq_count--;
    pthread_mutex_unlock(&ctx->fq_mutex);
    free(node);
    return 1;
}

void jm_decoder_flush(JMDecoderContext *ctx)
{
    if (!ctx || !ctx->mt)
        return;
    fprintf(stderr, "[BRIDGE] jm_decoder_flush: calling FlushDecoderMT\n");
    FlushDecoderMT(ctx->mt);
    fprintf(stderr, "[BRIDGE] jm_decoder_flush: FlushDecoderMT done\n");
    ctx->finished = 1;
    pthread_mutex_lock(&ctx->fq_mutex);
    pthread_cond_signal(&ctx->fq_cond);
    pthread_mutex_unlock(&ctx->fq_mutex);
}

int jm_decoder_finished(JMDecoderContext *ctx)
{
    if (!ctx) return 1;
    if (ctx->finished) return 1;
    /* Also check if MT decoder threads have finished on their own */
    if (ctx->mt && ctx->mt->view_finished[0] && ctx->mt->view_finished[1])
        return 1;
    return 0;
}

void jm_frame_free(JMDecodedFrame *frame)
{
    if (!frame)
        return;
    free(frame->planes[0]);
    free(frame->planes[1]);
    free(frame->planes[2]);
    frame->planes[0] = frame->planes[1] = frame->planes[2] = NULL;
}

void jm_decoder_close(JMDecoderContext *ctx)
{
    int i;
    FrameNode *node, *next;
    if (!ctx)
        return;
    if (ctx->mt) {
        CloseDecoderMT(ctx->mt);
        ctx->mt = NULL;
    }
    /* Free any remaining per-view FIFO entries */
    for (i = 0; i < 2; i++) {
        ViewFrameNode *vn = ctx->view_head[i];
        while (vn) {
            ViewFrameNode *vnext = vn->next;
            jm_frame_free(&vn->frame);
            free(vn);
            vn = vnext;
        }
    }
    /* Free remaining queued frames */
    node = ctx->fq_head;
    while (node) {
        next = node->next;
        jm_frame_free(&node->frame);
        free(node);
        node = next;
    }
    pthread_mutex_destroy(&ctx->fq_mutex);
    pthread_cond_destroy(&ctx->fq_cond);
    pthread_mutex_destroy(&ctx->pair_mutex);
    free(ctx);
}
