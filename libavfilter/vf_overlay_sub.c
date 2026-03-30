/*
 * Subtitle overlay filter for both 2D and side-by-side 3D video.
 *
 * Composites RGBA subtitle frames (from sub2video) onto YUV420P video.
 * In SBS 3D mode, composites onto both eye halves with an optional
 * horizontal pixel offset for 3D depth (read from OFMD SEI side data).
 * In 2D mode, composites once onto the full frame.
 *
 * Mode is auto-detected from video vs subtitle dimensions, or can be
 * forced via the "mode" option.
 *
 * Copyright (C) 2025
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License,
 * or (at your option) any later version.
 */

#include <math.h>
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/csp.h"
#include "avfilter.h"
#include "colorspace.h"
#include "filters.h"
#include "formats.h"
#include "framesync.h"

enum OverlaySubMode {
    MODE_AUTO = 0,
    MODE_SBS  = 1,
    MODE_2D   = 2,
};

typedef struct OverlaySubContext {
    const AVClass *class;
    FFFrameSync fs;

    int mode;               /* user-selected mode (auto/sbs/2d) */
    int plane;              /* OFMD plane index override (-1 = auto from metadata) */
    int fallback_offset;    /* offset when no OFMD side data present */
    int sub_brightness;     /* subtitle brightness: -100 to +100, 0 = default */

    int video_width;        /* full video width */
    int eye_width;          /* blending width (full for 2D, half for SBS) */
    int eye_height;
    int hsub, vsub;         /* chroma subsampling of main video */

    int detected_mode;      /* resolved mode after first subtitle */
    int mode_detected;      /* whether detection has happened */
    int auto_plane;         /* plane index detected from subtitle metadata */
    int auto_plane_set;     /* whether auto_plane has been set */

    /* Color properties (detected on first frame) */
    int color_detected;     /* whether color detection has run */
    int is_hdr;             /* video uses PQ transfer function */
    int bit_depth;          /* 8 or 10 */
    double rgb2yuv[3][3];   /* color-space-aware RGB→YUV matrix */
    float brightness_scale; /* 1.0 + sub_brightness / 100.0 */
} OverlaySubContext;

#define OFFSET(x) offsetof(OverlaySubContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption overlay_sub_options[] = {
    { "mode", "overlay mode",
      OFFSET(mode), AV_OPT_TYPE_INT, { .i64 = MODE_AUTO }, MODE_AUTO, MODE_2D, FLAGS, .unit = "mode" },
        { "auto", "auto-detect from dimensions", 0, AV_OPT_TYPE_CONST, { .i64 = MODE_AUTO }, .flags = FLAGS, .unit = "mode" },
        { "sbs",  "side-by-side 3D",             0, AV_OPT_TYPE_CONST, { .i64 = MODE_SBS },  .flags = FLAGS, .unit = "mode" },
        { "2d",   "standard 2D overlay",         0, AV_OPT_TYPE_CONST, { .i64 = MODE_2D },   .flags = FLAGS, .unit = "mode" },
    { "plane", "OFMD plane index (-1 = auto from 3d-plane metadata)",
      OFFSET(plane), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 31, FLAGS },
    { "fallback_offset", "pixel offset when no OFMD data present",
      OFFSET(fallback_offset), AV_OPT_TYPE_INT, { .i64 = 0 }, -127, 127, FLAGS },
    { "sub_brightness", "subtitle brightness adjustment (-100 to 100)",
      OFFSET(sub_brightness), AV_OPT_TYPE_INT, { .i64 = 0 }, -100, 100, FLAGS },
    { "eof_action", "action on subtitle EOF",
      OFFSET(fs.opt_eof_action), AV_OPT_TYPE_INT, { .i64 = EOF_ACTION_PASS },
      EOF_ACTION_REPEAT, EOF_ACTION_PASS, FLAGS, .unit = "eof_action" },
        { "repeat", "Repeat the previous frame",   0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_REPEAT }, .flags = FLAGS, .unit = "eof_action" },
        { "endall", "End both streams",             0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_ENDALL }, .flags = FLAGS, .unit = "eof_action" },
        { "pass",   "Pass through the main input",  0, AV_OPT_TYPE_CONST, { .i64 = EOF_ACTION_PASS },   .flags = FLAGS, .unit = "eof_action" },
    { "shortest", "force shortest", OFFSET(fs.opt_shortest),
      AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(overlay_sub, OverlaySubContext, fs);

/* ---- Read OFMD offset from video frame side data ---- */

static int get_ofmd_offset(const AVFrame *video, int plane_idx)
{
    int i;
    for (i = 0; i < video->nb_side_data; i++) {
        const AVFrameSideData *sd = video->side_data[i];
        if (sd->type == AV_FRAME_DATA_SEI_UNREGISTERED &&
            sd->size >= 18 &&
            sd->data[0] == 'O' && sd->data[1] == 'F' &&
            sd->data[2] == 'M' && sd->data[3] == 'D') {
            int num_planes = sd->data[16];
            int swap_eyes  = sd->data[17];
            if (plane_idx >= 0 && plane_idx < num_planes &&
                sd->size >= 18 + num_planes) {
                int off = (int)(int8_t)sd->data[18 + plane_idx];
                return swap_eyes ? -off : off;
            }
            return 0;
        }
    }
    return 0;  /* no OFMD data */
}

/* ---- Detect color properties on first frame ---- */

static void detect_color_properties(OverlaySubContext *s, const AVFrame *video,
                                    AVFilterContext *ctx)
{
    const AVPixFmtDescriptor *desc;
    const AVLumaCoefficients *coeffs;

    if (s->color_detected)
        return;

    desc = av_pix_fmt_desc_get(video->format);
    s->bit_depth = desc->comp[0].depth;
    s->is_hdr = (video->color_trc == AVCOL_TRC_SMPTE2084);
    s->brightness_scale = 1.0f + s->sub_brightness / 100.0f;

    /* Build RGB→YUV matrix from video's color space */
    coeffs = av_csp_luma_coeffs_from_avcsp(video->colorspace);
    if (coeffs)
        ff_fill_rgb2yuv_table(coeffs, s->rgb2yuv);
    else
        ff_fill_rgb2yuv_table(av_csp_luma_coeffs_from_avcsp(AVCOL_SPC_BT709), s->rgb2yuv);

    av_log(ctx, AV_LOG_INFO, "overlay_sub: color detection: %s, %d-bit, colorspace=%d, brightness=%+d%%\n",
           s->is_hdr ? "HDR10 (PQ)" : "SDR", s->bit_depth, video->colorspace, s->sub_brightness);

    s->color_detected = 1;
}

/* ---- Composite RGBA subtitle onto YUV frame (8-bit SDR fast path) ---- */

static void blend_subtitle_8bit(AVFrame *dst, const AVFrame *sub,
                                int eye_x_offset, int h_offset,
                                int eye_width, int eye_height,
                                int hsub, int vsub,
                                float brightness_scale)
{
    int x, y;
    int sub_w = sub->width;
    int sub_h = sub->height;
    const uint8_t *src_row;
    int dst_stride_y = dst->linesize[0];
    int dst_stride_u = dst->linesize[1];
    int dst_stride_v = dst->linesize[2];
    int src_stride   = sub->linesize[0];
    int use_brightness = (brightness_scale < 0.999f || brightness_scale > 1.001f);

    for (y = 0; y < sub_h && y < eye_height; y++) {
        src_row = sub->data[0] + y * src_stride;

        for (x = 0; x < sub_w; x++) {
            uint8_t r = src_row[x * 4 + 0];
            uint8_t g = src_row[x * 4 + 1];
            uint8_t b = src_row[x * 4 + 2];
            uint8_t a = src_row[x * 4 + 3];
            int Y, U, V;

            if (a == 0)
                continue;

            int dx = x + h_offset;
            if (dx < 0 || dx >= eye_width)
                continue;

            if (use_brightness) {
                /* Apply brightness in linear light, then BT.709 */
                float rf = powf(r / 255.0f, 2.4f) * brightness_scale;
                float gf = powf(g / 255.0f, 2.4f) * brightness_scale;
                float bf = powf(b / 255.0f, 2.4f) * brightness_scale;
                rf = FFMIN(FFMAX(rf, 0.0f), 1.0f);
                gf = FFMIN(FFMAX(gf, 0.0f), 1.0f);
                bf = FFMIN(FFMAX(bf, 0.0f), 1.0f);
                /* Back to gamma */
                rf = powf(rf, 1.0f / 2.4f);
                gf = powf(gf, 1.0f / 2.4f);
                bf = powf(bf, 1.0f / 2.4f);
                r = (uint8_t)(rf * 255.0f + 0.5f);
                g = (uint8_t)(gf * 255.0f + 0.5f);
                b = (uint8_t)(bf * 255.0f + 0.5f);
            }

            /* Convert RGB to YUV (BT.709) */
            Y = (( 47 * r + 157 * g +  16 * b + 128) >> 8) + 16;
            U = ((-26 * r -  87 * g + 112 * b + 128) >> 8) + 128;
            V = ((112 * r -  102 * g -  10 * b + 128) >> 8) + 128;

            /* Alpha blend Y */
            int dst_idx = eye_x_offset + dx;
            uint8_t *py = dst->data[0] + y * dst_stride_y + dst_idx;
            *py = (uint8_t)((*py * (255 - a) + Y * a + 127) / 255);

            /* Alpha blend U/V (chroma subsampled) */
            if ((y & ((1 << vsub) - 1)) == 0 && (dx & ((1 << hsub) - 1)) == 0) {
                int cx = dst_idx >> hsub;
                int cy = y >> vsub;
                uint8_t *pu = dst->data[1] + cy * dst_stride_u + cx;
                uint8_t *pv = dst->data[2] + cy * dst_stride_v + cx;
                *pu = (uint8_t)((*pu * (255 - a) + U * a + 127) / 255);
                *pv = (uint8_t)((*pv * (255 - a) + V * a + 127) / 255);
            }
        }
    }
}

/* ---- Composite RGBA subtitle onto 10-bit HDR YUV frame ---- */

static void blend_subtitle_10bit(AVFrame *dst, const AVFrame *sub,
                                 int eye_x_offset, int h_offset,
                                 int eye_width, int eye_height,
                                 int hsub, int vsub,
                                 const OverlaySubContext *s)
{
    int x, y;
    int sub_w = sub->width;
    int sub_h = sub->height;
    const uint8_t *src_row;
    int dst_stride_y = dst->linesize[0] / 2;  /* stride in uint16_t units */
    int dst_stride_u = dst->linesize[1] / 2;
    int dst_stride_v = dst->linesize[2] / 2;
    int src_stride   = sub->linesize[0];
    float ref_white;

    /* Compute adjusted reference white for HDR brightness.
     * Use 203 nits (ITU-R BT.2408 SDR reference white for HDR displays)
     * as the baseline for mapping SDR subtitle colors into PQ space. */
    if (s->is_hdr) {
        ref_white = REFERENCE_WHITE_ALT * s->brightness_scale;
        ref_white = FFMIN(FFMAX(ref_white, 10.0f), 600.0f);
    } else {
        ref_white = REFERENCE_WHITE_ALT;
    }

    for (y = 0; y < sub_h && y < eye_height; y++) {
        src_row = sub->data[0] + y * src_stride;

        for (x = 0; x < sub_w; x++) {
            uint8_t sr = src_row[x * 4 + 0];
            uint8_t sg = src_row[x * 4 + 1];
            uint8_t sb = src_row[x * 4 + 2];
            uint8_t a  = src_row[x * 4 + 3];
            float rf, gf, bf, Y_f, Cb_f, Cr_f;
            int Y_out, Cb_out, Cr_out;

            if (a == 0)
                continue;

            int dx = x + h_offset;
            if (dx < 0 || dx >= eye_width)
                continue;

            /* Linearize SDR subtitle (BT.1886 gamma ≈ 2.4) */
            rf = powf(sr / 255.0f, 2.4f);
            gf = powf(sg / 255.0f, 2.4f);
            bf = powf(sb / 255.0f, 2.4f);

            if (!s->is_hdr) {
                /* 10-bit SDR: apply brightness in linear, back to gamma, matrix */
                rf *= s->brightness_scale;
                gf *= s->brightness_scale;
                bf *= s->brightness_scale;
                rf = FFMIN(FFMAX(rf, 0.0f), 1.0f);
                gf = FFMIN(FFMAX(gf, 0.0f), 1.0f);
                bf = FFMIN(FFMAX(bf, 0.0f), 1.0f);
                rf = powf(rf, 1.0f / 2.4f);
                gf = powf(gf, 1.0f / 2.4f);
                bf = powf(bf, 1.0f / 2.4f);
            } else {
                /* HDR: encode linear light to PQ at adjusted reference white */
                rf = inverse_eotf_st2084(rf, ref_white);
                gf = inverse_eotf_st2084(gf, ref_white);
                bf = inverse_eotf_st2084(bf, ref_white);
            }

            /* RGB→YUV using detected color space matrix */
            Y_f  = s->rgb2yuv[0][0] * rf + s->rgb2yuv[0][1] * gf + s->rgb2yuv[0][2] * bf;
            Cb_f = s->rgb2yuv[1][0] * rf + s->rgb2yuv[1][1] * gf + s->rgb2yuv[1][2] * bf;
            Cr_f = s->rgb2yuv[2][0] * rf + s->rgb2yuv[2][1] * gf + s->rgb2yuv[2][2] * bf;

            /* Quantize to 10-bit limited range */
            Y_out  = (int)(Y_f  * 876.0f + 0.5f) + 64;   /* 219*4=876, offset=16*4=64 */
            Cb_out = (int)(Cb_f * 896.0f + 0.5f) + 512;   /* 224*4=896, offset=128*4=512 */
            Cr_out = (int)(Cr_f * 896.0f + 0.5f) + 512;
            Y_out  = FFMIN(FFMAX(Y_out, 64), 940);
            Cb_out = FFMIN(FFMAX(Cb_out, 64), 960);
            Cr_out = FFMIN(FFMAX(Cr_out, 64), 960);

            /* Alpha blend Y (10-bit) */
            int dst_idx = eye_x_offset + dx;
            uint16_t *py = (uint16_t *)dst->data[0] + y * dst_stride_y + dst_idx;
            *py = (uint16_t)((*py * (255 - a) + Y_out * a + 127) / 255);

            /* Alpha blend U/V (chroma subsampled) */
            if ((y & ((1 << vsub) - 1)) == 0 && (dx & ((1 << hsub) - 1)) == 0) {
                int cx = dst_idx >> hsub;
                int cy = y >> vsub;
                uint16_t *pu = (uint16_t *)dst->data[1] + cy * dst_stride_u + cx;
                uint16_t *pv = (uint16_t *)dst->data[2] + cy * dst_stride_v + cx;
                *pu = (uint16_t)((*pu * (255 - a) + Cb_out * a + 127) / 255);
                *pv = (uint16_t)((*pv * (255 - a) + Cr_out * a + 127) / 255);
            }
        }
    }
}

/* ---- Dispatcher: choose blend path based on bit depth ---- */

static void blend_subtitle_onto_eye(AVFrame *dst, const AVFrame *sub,
                                    int eye_x_offset, int h_offset,
                                    int eye_width, int eye_height,
                                    int hsub, int vsub,
                                    const OverlaySubContext *s)
{
    if (s->bit_depth > 8) {
        blend_subtitle_10bit(dst, sub, eye_x_offset, h_offset,
                             eye_width, eye_height, hsub, vsub, s);
    } else {
        blend_subtitle_8bit(dst, sub, eye_x_offset, h_offset,
                            eye_width, eye_height, hsub, vsub,
                            s->brightness_scale);
    }
}

/* ---- Mode detection ---- */

static void detect_mode(OverlaySubContext *s, const AVFrame *video,
                        const AVFrame *sub, AVFilterContext *ctx)
{
    if (s->mode_detected)
        return;

    if (s->mode == MODE_SBS) {
        s->detected_mode = MODE_SBS;
        s->eye_width = s->video_width / 2;
        av_log(ctx, AV_LOG_INFO, "overlay_sub: forced SBS mode, eye=%dx%d\n",
               s->eye_width, s->eye_height);
    } else if (s->mode == MODE_2D) {
        s->detected_mode = MODE_2D;
        av_log(ctx, AV_LOG_INFO, "overlay_sub: forced 2D mode, %dx%d\n",
               s->eye_width, s->eye_height);
    } else {
        /* Auto-detect: SBS if video width >= 2x subtitle width (with tolerance) */
        if (sub->width > 0 && s->video_width >= sub->width * 2 - 16) {
            s->detected_mode = MODE_SBS;
            s->eye_width = s->video_width / 2;
            av_log(ctx, AV_LOG_INFO,
                   "overlay_sub: auto-detected SBS mode (video %dx%d, sub %dx%d), eye=%dx%d\n",
                   s->video_width, s->eye_height, sub->width, sub->height,
                   s->eye_width, s->eye_height);
        } else {
            s->detected_mode = MODE_2D;
            av_log(ctx, AV_LOG_INFO,
                   "overlay_sub: auto-detected 2D mode (video %dx%d, sub %dx%d)\n",
                   s->video_width, s->eye_height, sub->width, sub->height);
        }
    }
    s->mode_detected = 1;
}

/* ---- Framesync callback ---- */

static int process_frame(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    OverlaySubContext *s = ctx->priv;
    AVFrame *video, *sub;
    int ret;

    ret = ff_framesync_dualinput_get_writable(fs, &video, &sub);
    if (ret < 0)
        return ret;

    if (!sub)
        return ff_filter_frame(ctx->outputs[0], video);

    /* Detect mode and color properties on first subtitle frame */
    detect_mode(s, video, sub, ctx);
    detect_color_properties(s, video, ctx);

    if (s->detected_mode == MODE_2D) {
        /* 2D mode: single blend, full width, no offset */
        blend_subtitle_onto_eye(video, sub, 0, 0,
                                s->eye_width, s->eye_height,
                                s->hsub, s->vsub, s);
    } else {
        /* SBS mode: blend both eyes with OFMD depth offset */
        int offset, plane_idx;

        /* Determine plane index */
        plane_idx = s->plane;
        if (plane_idx < 0) {
            if (!s->auto_plane_set) {
                const AVDictionaryEntry *e = av_dict_get(sub->metadata, "3d-plane", NULL, 0);
                if (e) {
                    s->auto_plane = atoi(e->value);
                    av_log(ctx, AV_LOG_INFO, "overlay_sub: auto-detected 3d-plane=%d\n", s->auto_plane);
                } else {
                    s->auto_plane = 0;
                    av_log(ctx, AV_LOG_WARNING, "overlay_sub: no 3d-plane metadata, defaulting to plane 0\n");
                }
                s->auto_plane_set = 1;
            }
            plane_idx = s->auto_plane;
        }

        /* Get per-frame offset from OFMD side data */
        offset = get_ofmd_offset(video, plane_idx);
        av_log(ctx, AV_LOG_DEBUG, "overlay_sub: frame pts=%"PRId64" plane=%d ofmd_offset=%d sub=%dx%d\n",
               video->pts, plane_idx, offset, sub->width, sub->height);
        if (offset == 0)
            offset = s->fallback_offset;

        /* Composite subtitle onto left eye (x=0, no shift) */
        blend_subtitle_onto_eye(video, sub, 0, 0,
                                s->eye_width, s->eye_height,
                                s->hsub, s->vsub, s);

        /* Composite subtitle onto right eye (x=eye_width, with depth offset) */
        blend_subtitle_onto_eye(video, sub, s->eye_width, offset,
                                s->eye_width, s->eye_height,
                                s->hsub, s->vsub, s);
    }

    return ff_filter_frame(ctx->outputs[0], video);
}

/* ---- Filter lifecycle ---- */

static av_cold int overlay_sub_init(AVFilterContext *ctx)
{
    OverlaySubContext *s = ctx->priv;
    s->fs.on_event = process_frame;
    return 0;
}

static av_cold void overlay_sub_uninit(AVFilterContext *ctx)
{
    OverlaySubContext *s = ctx->priv;
    ff_framesync_uninit(&s->fs);
}

static int config_input_main(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    OverlaySubContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;
    s->video_width = inlink->w;
    s->eye_width = inlink->w;  /* default to full width; halved if SBS detected */
    s->eye_height = inlink->h;

    av_log(ctx, AV_LOG_INFO, "overlay_sub: video %dx%d\n", inlink->w, inlink->h);
    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    OverlaySubContext *s = ctx->priv;
    int ret;

    if ((ret = ff_framesync_init_dualinput(&s->fs, ctx)) < 0)
        return ret;

    outlink->w = ctx->inputs[0]->w;
    outlink->h = ctx->inputs[0]->h;
    outlink->time_base = ctx->inputs[0]->time_base;

    return ff_framesync_configure(&s->fs);
}

static int overlay_sub_activate(AVFilterContext *ctx)
{
    OverlaySubContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    /* Main input: YUV420P or YUV420P10LE */
    static const enum AVPixelFormat main_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_NONE
    };
    /* Subtitle input: RGBA (from sub2video) */
    static const enum AVPixelFormat sub_fmts[] = {
        AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE
    };
    /* Output: same as main input */
    int ret;
    AVFilterFormats *fmts;

    fmts = ff_make_format_list(main_fmts);
    if ((ret = ff_formats_ref(fmts, &cfg_in[0]->formats)) < 0)
        return ret;
    if ((ret = ff_formats_ref(ff_make_format_list(sub_fmts),
                              &cfg_in[1]->formats)) < 0)
        return ret;
    if ((ret = ff_formats_ref(fmts, &cfg_out[0]->formats)) < 0)
        return ret;
    return 0;
}

/* ---- Pads ---- */

static const AVFilterPad overlay_sub_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_main,
    },
    {
        .name         = "overlay",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
};

static const AVFilterPad overlay_sub_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
};

const AVFilter ff_vf_overlay_sub = {
    .name           = "overlay_sub",
    .description    = NULL_IF_CONFIG_SMALL("Overlay subtitles on video (2D or side-by-side 3D with depth offset)."),
    .preinit        = overlay_sub_framesync_preinit,
    .init           = overlay_sub_init,
    .uninit         = overlay_sub_uninit,
    .priv_size      = sizeof(OverlaySubContext),
    .priv_class     = &overlay_sub_class,
    .activate       = overlay_sub_activate,
    FILTER_INPUTS(overlay_sub_inputs),
    FILTER_OUTPUTS(overlay_sub_outputs),
    FILTER_QUERY_FUNC2(query_formats),
    .flags          = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};
