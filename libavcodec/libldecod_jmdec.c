/*
 * JM reference H.264/MVC decoder wrapper for FFmpeg
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

#include <string.h>
#include <unistd.h>

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/stereo3d.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "packet.h"

#include "libldecod_jm_bridge.h"

#define PTS_QUEUE_SIZE 128

typedef struct JMDecContext {
    AVClass *av_class;
    JMDecoderContext *jm;
    int mvc_output;
    int swap_eyes;

    int64_t pts_queue[PTS_QUEUE_SIZE];
    int pts_count;

    int initialized;
    int eof_sent;
} JMDecContext;

static void insert_pts_sorted(JMDecContext *ctx, int64_t pts)
{
    int i;
    if (pts == AV_NOPTS_VALUE || ctx->pts_count >= PTS_QUEUE_SIZE)
        return;
    i = ctx->pts_count;
    while (i > 0 && ctx->pts_queue[i - 1] > pts) {
        ctx->pts_queue[i] = ctx->pts_queue[i - 1];
        i--;
    }
    ctx->pts_queue[i] = pts;
    ctx->pts_count++;
}

static int64_t pop_smallest_pts(JMDecContext *ctx)
{
    int64_t pts;
    int i;
    if (ctx->pts_count == 0)
        return AV_NOPTS_VALUE;
    pts = ctx->pts_queue[0];
    for (i = 1; i < ctx->pts_count; i++)
        ctx->pts_queue[i - 1] = ctx->pts_queue[i];
    ctx->pts_count--;
    return pts;
}

static av_cold int jmdec_init(AVCodecContext *avctx)
{
    JMDecContext *ctx = avctx->priv_data;
    int i;

    ctx->jm = jm_decoder_open(1);
    if (!ctx->jm) {
        av_log(avctx, AV_LOG_ERROR, "Failed to open JM decoder\n");
        return AVERROR(ENOMEM);
    }

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->pts_count = 0;
    ctx->swap_eyes = 0;

    for (i = 0; i < avctx->nb_coded_side_data; i++) {
        if (avctx->coded_side_data[i].type == AV_PKT_DATA_STEREO3D) {
            const AVStereo3D *stereo = (const AVStereo3D *)avctx->coded_side_data[i].data;
            if (!(stereo->flags & AV_STEREO3D_FLAG_INVERT)) {
                ctx->swap_eyes = 1;
                av_log(avctx, AV_LOG_DEBUG, "Detected block_lr stereo mode, swapping eyes\n");
            }
            break;
        }
    }

    ctx->initialized = 1;
    av_log(avctx, AV_LOG_INFO, "JM H.264/MVC decoder initialized (mvc_output=%d)\n",
           ctx->mvc_output);
    return 0;
}

static av_cold int jmdec_close(AVCodecContext *avctx)
{
    JMDecContext *ctx = avctx->priv_data;
    if (ctx->jm)
        jm_decoder_close(ctx->jm);
    ctx->jm = NULL;
    return 0;
}

static av_cold void jmdec_flush(AVCodecContext *avctx)
{
    JMDecContext *ctx = avctx->priv_data;
    ctx->pts_count = 0;
}

static int output_jm_frame(AVCodecContext *avctx, AVFrame *avframe,
                           JMDecodedFrame *jmframe, JMDecContext *ctx)
{
    int ret, y;
    int w = jmframe->width;
    int h = jmframe->height;
    int src_stride_y, src_stride_uv;
    int hw, hh;
    AVStereo3D *stereo;

    if (ctx->mvc_output && jmframe->is_mvc)
        ret = ff_set_dimensions(avctx, w * 2, h);
    else
        ret = ff_set_dimensions(avctx, w, h);
    if (ret < 0)
        return ret;

    ret = ff_get_buffer(avctx, avframe, 0);
    if (ret < 0)
        return ret;

    if (ctx->mvc_output && jmframe->is_mvc) {
        src_stride_y = jmframe->strides[0];
        src_stride_uv = jmframe->strides[1];
        hw = w / 2;
        hh = h / 2;

        if (ctx->swap_eyes) {
            for (y = 0; y < h; y++) {
                memcpy(avframe->data[0] + y * avframe->linesize[0],
                       jmframe->planes[0] + y * src_stride_y + w, w);
                memcpy(avframe->data[0] + y * avframe->linesize[0] + w,
                       jmframe->planes[0] + y * src_stride_y, w);
            }
            if (jmframe->planes[1]) {
                for (y = 0; y < hh; y++) {
                    memcpy(avframe->data[1] + y * avframe->linesize[1],
                           jmframe->planes[1] + y * src_stride_uv + hw, hw);
                    memcpy(avframe->data[1] + y * avframe->linesize[1] + hw,
                           jmframe->planes[1] + y * src_stride_uv, hw);
                }
                for (y = 0; y < hh; y++) {
                    memcpy(avframe->data[2] + y * avframe->linesize[2],
                           jmframe->planes[2] + y * src_stride_uv + hw, hw);
                    memcpy(avframe->data[2] + y * avframe->linesize[2] + hw,
                           jmframe->planes[2] + y * src_stride_uv, hw);
                }
            }
        } else {
            for (y = 0; y < h; y++)
                memcpy(avframe->data[0] + y * avframe->linesize[0],
                       jmframe->planes[0] + y * src_stride_y, w * 2);
            if (jmframe->planes[1]) {
                for (y = 0; y < hh; y++)
                    memcpy(avframe->data[1] + y * avframe->linesize[1],
                           jmframe->planes[1] + y * src_stride_uv, w);
                for (y = 0; y < hh; y++)
                    memcpy(avframe->data[2] + y * avframe->linesize[2],
                           jmframe->planes[2] + y * src_stride_uv, w);
            }
        }

        stereo = av_stereo3d_create_side_data(avframe);
        if (stereo)
            stereo->type = AV_STEREO3D_SIDEBYSIDE;
    } else {
        hw = w / 2;
        hh = h / 2;
        for (y = 0; y < h; y++)
            memcpy(avframe->data[0] + y * avframe->linesize[0],
                   jmframe->planes[0] + y * jmframe->strides[0], w);
        if (jmframe->planes[1]) {
            for (y = 0; y < hh; y++)
                memcpy(avframe->data[1] + y * avframe->linesize[1],
                       jmframe->planes[1] + y * jmframe->strides[1], hw);
            for (y = 0; y < hh; y++)
                memcpy(avframe->data[2] + y * avframe->linesize[2],
                       jmframe->planes[2] + y * jmframe->strides[2], hw);
        }
    }

    avframe->pts = pop_smallest_pts(ctx);
    return 0;
}

static int jmdec_receive_frame(AVCodecContext *avctx, AVFrame *avframe)
{
    static const uint8_t start_code[] = {0, 0, 0, 1};
    JMDecContext *ctx = avctx->priv_data;
    JMDecodedFrame jmframe;
    AVPacket *pkt;
    int ret;
    size_t side_size;
    const uint8_t *side_data;
    const uint8_t *p, *end;
    uint32_t nalu_len;

    /* Check for already-decoded frames first (non-blocking) */
    if (jm_decoder_get_frame(ctx->jm, &jmframe)) {
        ret = output_jm_frame(avctx, avframe, &jmframe, ctx);
        jm_frame_free(&jmframe);
        return ret;
    }

    /* Draining mode: block-wait for frames from decoder thread */
    if (ctx->eof_sent) {
        if (jm_decoder_get_frame_blocking(ctx->jm, &jmframe)) {
            ret = output_jm_frame(avctx, avframe, &jmframe, ctx);
            jm_frame_free(&jmframe);
            return ret;
        }
        return AVERROR_EOF;
    }

    /* Feed ONE packet, then return EAGAIN so FFmpeg calls us again
     * (which will check for frames first, preventing deadlock) */
    pkt = av_packet_alloc();
    if (!pkt)
        return AVERROR(ENOMEM);

    ret = ff_decode_get_packet(avctx, pkt);
    if (ret < 0) {
        av_packet_free(&pkt);
        if (ret == AVERROR_EOF) {
            jm_decoder_flush(ctx->jm);
            ctx->eof_sent = 1;
            /* Try to get a frame immediately after signaling EOF */
            if (jm_decoder_get_frame_blocking(ctx->jm, &jmframe)) {
                ret = output_jm_frame(avctx, avframe, &jmframe, ctx);
                jm_frame_free(&jmframe);
                return ret;
            }
            return AVERROR_EOF;
        }
        return ret;
    }

    insert_pts_sorted(ctx, pkt->pts);

    /* Feed main packet data (already Annex B due to BSF) */
    if (pkt->data && pkt->size > 0)
        jm_decoder_feed(ctx->jm, pkt->data, pkt->size);

    /* Feed MVC side data (BlockAdditional) */
    side_size = 0;
    side_data = av_packet_get_side_data(pkt,
        AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL, &side_size);
    if (side_data && side_size > 8) {
        p = side_data + 8;
        end = side_data + side_size;
        while (p + 4 <= end) {
            nalu_len = AV_RB32(p);
            p += 4;
            if (p + nalu_len > end)
                break;
            jm_decoder_feed(ctx->jm, start_code, 4);
            jm_decoder_feed(ctx->jm, p, nalu_len);
            p += nalu_len;
        }
    }

    av_packet_free(&pkt);

    /* After feeding, check for a frame before returning */
    if (jm_decoder_get_frame(ctx->jm, &jmframe)) {
        ret = output_jm_frame(avctx, avframe, &jmframe, ctx);
        jm_frame_free(&jmframe);
        return ret;
    }

    return AVERROR(EAGAIN);
}

#define OFFSET(x) offsetof(JMDecContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM

static const AVOption jmdec_options[] = {
    { "mvc_output", "Output MVC as side-by-side", OFFSET(mvc_output),
      AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VD },
    { NULL }
};

static const AVClass libldecod_jm_decoder_class = {
    .class_name = "libldecod_jm decoder",
    .item_name  = av_default_item_name,
    .option     = jmdec_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libldecod_jm_decoder = {
    .p.name         = "libldecod_jm",
    CODEC_LONG_NAME("JM reference H.264 / MVC decoder"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .p.priv_class   = &libldecod_jm_decoder_class,
    .priv_data_size = sizeof(JMDecContext),
    .init           = jmdec_init,
    FF_CODEC_RECEIVE_FRAME_CB(jmdec_receive_frame),
    .close          = jmdec_close,
    .flush          = jmdec_flush,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .bsfs           = "h264_mp4toannexb",
    .p.wrapper_name = "libldecod_jm",
};
