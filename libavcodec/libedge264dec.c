/*
 * edge264 video decoder
 * Copyright (C) 2025
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <edge264.h>
#include <inttypes.h>
#include <string.h>

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/stereo3d.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "packet.h"

#define FRAME_QUEUE_SIZE 32
#define PTS_QUEUE_SIZE 64

typedef struct Edge264Context {
    AVClass *av_class;
    Edge264Decoder *decoder;
    int mvc_output;  // 0 = base view only, 1 = SBS output
    int swap_eyes;   // swap left/right eye order for SBS output

    // Frame queue for buffered output
    AVFrame *frame_queue[FRAME_QUEUE_SIZE];
    int32_t frame_id_queue[FRAME_QUEUE_SIZE];
    int queue_head;
    int queue_tail;

    // PTS queue - sorted in ascending order for display-order assignment
    int64_t pts_queue[PTS_QUEUE_SIZE];
    int pts_count;

    // DTS queue - FIFO order for decode-order assignment
    int64_t dts_queue[PTS_QUEUE_SIZE];
    int dts_count;
} Edge264Context;

static av_cold int edge264_decode_close(AVCodecContext *avctx)
{
    Edge264Context *ctx = avctx->priv_data;

    while (ctx->queue_head != ctx->queue_tail) {
        av_frame_free(&ctx->frame_queue[ctx->queue_head]);
        ctx->queue_head = (ctx->queue_head + 1) % FRAME_QUEUE_SIZE;
    }

    if (ctx->decoder)
        edge264_free(&ctx->decoder);

    return 0;
}

static av_cold void edge264_decode_flush(AVCodecContext *avctx)
{
    Edge264Context *ctx = avctx->priv_data;

    // Clear frame queue
    while (ctx->queue_head != ctx->queue_tail) {
        av_frame_free(&ctx->frame_queue[ctx->queue_head]);
        ctx->queue_head = (ctx->queue_head + 1) % FRAME_QUEUE_SIZE;
    }

    // Clear PTS and DTS queues
    ctx->pts_count = 0;
    ctx->dts_count = 0;

    // Flush decoder internal state - edge264 handles keyframe requirements internally
    if (ctx->decoder)
        edge264_flush(ctx->decoder);
}

static int queue_frame(Edge264Context *ctx, AVFrame *frame, int32_t frame_id)
{
    int next_tail = (ctx->queue_tail + 1) % FRAME_QUEUE_SIZE;
    if (next_tail == ctx->queue_head)
        return -1;
    ctx->frame_queue[ctx->queue_tail] = frame;
    ctx->frame_id_queue[ctx->queue_tail] = frame_id;
    ctx->queue_tail = next_tail;
    return 0;
}

static AVFrame *dequeue_frame(Edge264Context *ctx, int32_t *frame_id)
{
    if (ctx->queue_head == ctx->queue_tail)
        return NULL;
    AVFrame *frame = ctx->frame_queue[ctx->queue_head];
    *frame_id = ctx->frame_id_queue[ctx->queue_head];
    ctx->queue_head = (ctx->queue_head + 1) % FRAME_QUEUE_SIZE;
    return frame;
}

// Insert PTS into sorted queue (ascending order)
static void insert_pts_sorted(Edge264Context *ctx, int64_t pts)
{
    if (pts == AV_NOPTS_VALUE)
        return;
    if (ctx->pts_count >= PTS_QUEUE_SIZE)
        return;

    // Find insertion point (keep sorted ascending)
    int i = ctx->pts_count;
    while (i > 0 && ctx->pts_queue[i - 1] > pts) {
        ctx->pts_queue[i] = ctx->pts_queue[i - 1];
        i--;
    }
    ctx->pts_queue[i] = pts;
    ctx->pts_count++;
}

// Get and remove the smallest PTS (first in sorted queue)
static int64_t pop_smallest_pts(Edge264Context *ctx)
{
    if (ctx->pts_count == 0)
        return AV_NOPTS_VALUE;

    int64_t pts = ctx->pts_queue[0];
    // Shift remaining elements
    for (int i = 1; i < ctx->pts_count; i++)
        ctx->pts_queue[i - 1] = ctx->pts_queue[i];
    ctx->pts_count--;
    return pts;
}

// Add DTS to FIFO queue (decode order)
static void push_dts(Edge264Context *ctx, int64_t dts)
{
    if (dts == AV_NOPTS_VALUE)
        return;
    if (ctx->dts_count >= PTS_QUEUE_SIZE)
        return;
    ctx->dts_queue[ctx->dts_count++] = dts;
}

// Get and remove the oldest DTS (FIFO)
static int64_t pop_dts(Edge264Context *ctx)
{
    if (ctx->dts_count == 0)
        return AV_NOPTS_VALUE;

    int64_t dts = ctx->dts_queue[0];
    // Shift remaining elements
    for (int i = 1; i < ctx->dts_count; i++)
        ctx->dts_queue[i - 1] = ctx->dts_queue[i];
    ctx->dts_count--;
    return dts;
}

static void edge264_log_callback(const char *str, void *log_arg)
{
    AVCodecContext *avctx = log_arg;
    av_log(avctx, AV_LOG_DEBUG, "%s", str);
}

static int parse_avcc_mvcc(AVCodecContext *avctx, Edge264Decoder *decoder,
                           const uint8_t *data, int size)
{
    if (size < 7)
        return 0;

    if (data[0] != 1)
        return 0;

    int num_sps = data[5] & 0x1f;
    int offset = 6;

    for (int i = 0; i < num_sps && offset + 2 <= size; i++) {
        int sps_len = AV_RB16(data + offset);
        offset += 2;
        if (offset + sps_len > size)
            break;
        av_log(avctx, AV_LOG_DEBUG, "Decoding SPS from extradata (%d bytes)\n", sps_len);
        edge264_decode_NAL(decoder, data + offset, data + offset + sps_len, NULL, NULL);
        offset += sps_len;
    }

    if (offset < size) {
        int num_pps = data[offset] & 0xff;
        offset++;
        for (int i = 0; i < num_pps && offset + 2 <= size; i++) {
            int pps_len = AV_RB16(data + offset);
            offset += 2;
            if (offset + pps_len > size)
                break;
            av_log(avctx, AV_LOG_DEBUG, "Decoding PPS from extradata (%d bytes)\n", pps_len);
            edge264_decode_NAL(decoder, data + offset, data + offset + pps_len, NULL, NULL);
            offset += pps_len;
        }
    }

    return offset;
}

static av_cold int edge264_decode_init(AVCodecContext *avctx)
{
    Edge264Context *ctx = avctx->priv_data;

    ctx->decoder = edge264_alloc(0, edge264_log_callback, avctx, 0, NULL, NULL, NULL);
    if (!ctx->decoder) {
        av_log(avctx, AV_LOG_ERROR, "Unable to create edge264 decoder\n");
        return AVERROR(ENOMEM);
    }

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->pts_count = 0;
    ctx->dts_count = 0;
    ctx->swap_eyes = 0;

    // Check for stereo3d side data to determine eye order
    // block_rl (most 3D Blu-rays): base=right eye, no swap needed
    // block_lr: base=left eye, need to swap
    for (int i = 0; i < avctx->nb_coded_side_data; i++) {
        if (avctx->coded_side_data[i].type == AV_PKT_DATA_STEREO3D) {
            const AVStereo3D *stereo = (const AVStereo3D *)avctx->coded_side_data[i].data;
            // If NOT inverted, it's block_lr (left-right), so we need to swap
            // If inverted, it's block_rl (right-left), no swap needed
            if (!(stereo->flags & AV_STEREO3D_FLAG_INVERT)) {
                ctx->swap_eyes = 1;
                av_log(avctx, AV_LOG_DEBUG, "Detected block_lr stereo mode, swapping eyes\n");
            } else {
                av_log(avctx, AV_LOG_DEBUG, "Detected block_rl stereo mode, no swap needed\n");
            }
            break;
        }
    }

    if (avctx->extradata && avctx->extradata_size > 7) {
        const uint8_t *data = avctx->extradata;
        int size = avctx->extradata_size;

        int consumed = parse_avcc_mvcc(avctx, ctx->decoder, data, size);

        if (consumed > 0 && consumed < size) {
            av_log(avctx, AV_LOG_DEBUG, "Parsing mvcC at offset %d\n", consumed);
            parse_avcc_mvcc(avctx, ctx->decoder, data + consumed, size - consumed);
        }
    }

    return 0;
}

static int output_frame(AVCodecContext *avctx, AVFrame *avframe,
                        Edge264Frame *frame, Edge264Context *ctx)
{
    int ret;

    if (ctx->mvc_output && frame->samples_mvc[0]) {
        ret = ff_set_dimensions(avctx, frame->width_Y * 2, frame->height_Y);
    } else {
        ret = ff_set_dimensions(avctx, frame->width_Y, frame->height_Y);
    }
    if (ret < 0)
        return ret;

    ret = ff_get_buffer(avctx, avframe, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Unable to allocate buffer\n");
        return ret;
    }

    if (ctx->mvc_output && frame->samples_mvc[0]) {
        int w = frame->width_Y;
        int h = frame->height_Y;

        // Determine which view goes on which side
        // For block_rl (most Blu-rays): mvc=left eye, base=right eye, no swap
        // For block_lr: mvc=right eye, base=left eye, need swap
        const uint8_t *left_Y  = ctx->swap_eyes ? frame->samples[0]     : frame->samples_mvc[0];
        const uint8_t *right_Y = ctx->swap_eyes ? frame->samples_mvc[0] : frame->samples[0];
        const uint8_t *left_U  = ctx->swap_eyes ? frame->samples[1]     : frame->samples_mvc[1];
        const uint8_t *right_U = ctx->swap_eyes ? frame->samples_mvc[1] : frame->samples[1];
        const uint8_t *left_V  = ctx->swap_eyes ? frame->samples[2]     : frame->samples_mvc[2];
        const uint8_t *right_V = ctx->swap_eyes ? frame->samples_mvc[2] : frame->samples[2];

        for (int y = 0; y < h; y++) {
            memcpy(avframe->data[0] + y * avframe->linesize[0],
                   left_Y + y * frame->stride_Y, w);
            memcpy(avframe->data[0] + y * avframe->linesize[0] + w,
                   right_Y + y * frame->stride_Y, w);
        }
        for (int y = 0; y < h / 2; y++) {
            memcpy(avframe->data[1] + y * avframe->linesize[1],
                   left_U + y * frame->stride_C, w / 2);
            memcpy(avframe->data[1] + y * avframe->linesize[1] + w / 2,
                   right_U + y * frame->stride_C, w / 2);
        }
        for (int y = 0; y < h / 2; y++) {
            memcpy(avframe->data[2] + y * avframe->linesize[2],
                   left_V + y * frame->stride_C, w / 2);
            memcpy(avframe->data[2] + y * avframe->linesize[2] + w / 2,
                   right_V + y * frame->stride_C, w / 2);
        }
    } else {
        const uint8_t *src[4] = { frame->samples[0], frame->samples[1], frame->samples[2], NULL };
        int src_linesize[4] = { frame->stride_Y, frame->stride_C, frame->stride_C, 0 };

        // av_image_copy2 only reads from src; cast away const to match API signature
        av_image_copy2(avframe->data, avframe->linesize, (uint8_t * const *)src, src_linesize,
                       avctx->pix_fmt, avctx->width, avctx->height);
    }

    avframe->flags |= AV_FRAME_FLAG_KEY * (frame->FrameId == 0);

    return 0;
}

static int decode_nal_units_collect_frames(Edge264Decoder *decoder, const uint8_t *data, int size,
                                           AVCodecContext *avctx, Edge264Context *ctx)
{
    const uint8_t *nal = data;
    const uint8_t *end = data + size;
    int frame_count = 0;
    Edge264Frame frame;
    int ret;

    if (size >= 4 && nal[0] == 0 && nal[1] == 0) {
        if (nal[2] == 1) {
            nal += 3;
        } else if (nal[2] == 0 && nal[3] == 1) {
            nal += 4;
        }
    }

    while (nal < end) {
        const uint8_t *next_start = edge264_find_start_code(nal, end, 0);

        edge264_decode_NAL(decoder, nal, next_start, NULL, NULL);

        while (edge264_get_frame(decoder, &frame, 0) == 0) {
            AVFrame *new_frame = av_frame_alloc();
            if (!new_frame)
                continue;
            ret = output_frame(avctx, new_frame, &frame, ctx);
            if (ret < 0) {
                av_frame_free(&new_frame);
                continue;
            }
            queue_frame(ctx, new_frame, frame.FrameId);
            frame_count++;
        }

        nal = next_start;
        if (nal < end && nal[0] == 0 && nal[1] == 0) {
            if (nal[2] == 1) {
                nal += 3;
            } else if (nal + 3 < end && nal[2] == 0 && nal[3] == 1) {
                nal += 4;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return frame_count;
}

static int edge264_decode_frame(AVCodecContext *avctx, AVFrame *avframe,
                                int *got_frame, AVPacket *avpkt)
{
    Edge264Context *ctx = avctx->priv_data;
    Edge264Frame frame;
    int ret;

    *got_frame = 0;

    // Handle flush mode first (no packet data)
    if (!avpkt->data) {
        int32_t frame_id;
        AVFrame *queued = dequeue_frame(ctx, &frame_id);
        if (queued) {
            av_frame_move_ref(avframe, queued);
            av_frame_free(&queued);
            avframe->pts = pop_smallest_pts(ctx);
            *got_frame = 1;
            return 0;
        }
        // Drain decoder
        edge264_flush(ctx->decoder);
        while (edge264_get_frame(ctx->decoder, &frame, 0) == 0) {
            AVFrame *new_frame = av_frame_alloc();
            if (!new_frame)
                return AVERROR(ENOMEM);
            ret = output_frame(avctx, new_frame, &frame, ctx);
            if (ret < 0) {
                av_frame_free(&new_frame);
                return ret;
            }
            queue_frame(ctx, new_frame, frame.FrameId);
        }
        queued = dequeue_frame(ctx, &frame_id);
        if (queued) {
            av_frame_move_ref(avframe, queued);
            av_frame_free(&queued);
            avframe->pts = pop_smallest_pts(ctx);
            *got_frame = 1;
        }
        return 0;
    }

    // DECODE FIRST, then return frames.
    // This reduces output lag by processing the packet immediately rather than
    // returning old queued frames first. For A/V sync, video output should keep
    // pace with audio - returning stale frames first causes video to lag behind.

    // Add this packet's PTS and DTS to queues
    insert_pts_sorted(ctx, avpkt->pts);
    push_dts(ctx, avpkt->dts);

    // Decode NAL units FIRST
    decode_nal_units_collect_frames(ctx->decoder, avpkt->data, avpkt->size, avctx, ctx);

    // Check for MVC in side data
    size_t side_size = 0;
    const uint8_t *side_data = av_packet_get_side_data(avpkt,
        AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL, &side_size);
    if (side_data && side_size > 8) {
        decode_nal_units_collect_frames(ctx->decoder, side_data + 8, side_size - 8, avctx, ctx);
    }

    // NOW return a frame from the queue (includes frames just decoded)
    int32_t frame_id;
    AVFrame *queued = dequeue_frame(ctx, &frame_id);
    if (queued) {
        av_frame_move_ref(avframe, queued);
        av_frame_free(&queued);
        avframe->pts = pop_smallest_pts(ctx);
        *got_frame = 1;
    }

    return avpkt->size;
}

#define OFFSET(x) offsetof(Edge264Context, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM

static const AVOption options[] = {
    { "mvc_output", "Output MVC as side-by-side", OFFSET(mvc_output), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, VD },
    { NULL }
};

static const AVClass libedge264_decoder_class = {
    .class_name = "libedge264 decoder",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libedge264_decoder = {
    .p.name         = "libedge264",
    CODEC_LONG_NAME("edge264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (MVC)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .p.priv_class   = &libedge264_decoder_class,
    .priv_data_size = sizeof(Edge264Context),
    .init           = edge264_decode_init,
    FF_CODEC_DECODE_CB(edge264_decode_frame),
    .close          = edge264_decode_close,
    .flush          = edge264_decode_flush,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .bsfs           = "h264_mp4toannexb",
    .p.wrapper_name = "libedge264",
};
