/*
 * JM reference H.264/MVC decoder bridge - plain C types interface
 * No ldecod or FFmpeg headers included here.
 */

#ifndef LIBLDECOD_JM_BRIDGE_H
#define LIBLDECOD_JM_BRIDGE_H

#include <stdint.h>

#define JM_FRAME_QUEUE_SIZE 64

typedef struct JMDecoderContext JMDecoderContext;

typedef struct JMDecodedFrame {
    uint8_t *planes[3];    // Y, U, V (owned by this struct, caller must free with jm_frame_free)
    int width, height;     // single view dimensions
    int strides[3];        // Y, U, V strides
    int is_mvc;            // 1 = dual view, views side by side in planes
    int poc;               // picture order count

    // OFMD subtitle depth offsets (per-plane, for this frame)
    int8_t  ofs_offsets[32]; // offset value per 3D plane (-127 to +127 pixels)
    int     ofs_num_planes;  // number of valid planes (0 = no OFMD data)
} JMDecodedFrame;

// Open decoder. decode_all_layers=1 for MVC.
// Returns NULL on failure.
JMDecoderContext *jm_decoder_open(int decode_all_layers);

// Feed Annex B data into the decoder's ring buffer.
// This may block if the ring buffer is full.
// Returns bytes written (== size on success).
int jm_decoder_feed(JMDecoderContext *ctx, const uint8_t *data, int size);

// Non-blocking feed: write as much as fits, return bytes written.
// Returns 0 if ring buffer is full.
int jm_decoder_try_feed(JMDecoderContext *ctx, const uint8_t *data, int size);

// Check if a decoded frame is available (non-blocking).
// Returns 1 if a frame was retrieved, 0 if none available.
// The frame's planes must be freed with jm_frame_free() when done.
int jm_decoder_get_frame(JMDecoderContext *ctx, JMDecodedFrame *frame);

// Wait for a decoded frame (blocking).
// Returns 1 if a frame was retrieved, 0 if decoder finished with no more frames.
int jm_decoder_get_frame_blocking(JMDecoderContext *ctx, JMDecodedFrame *frame);

// Signal EOF - no more data will be fed.
// After this, keep calling jm_decoder_get_frame() to drain remaining frames.
void jm_decoder_flush(JMDecoderContext *ctx);

// Returns 1 if decoder thread has finished (all frames decoded after EOF).
int jm_decoder_finished(JMDecoderContext *ctx);

// Free frame plane data.
void jm_frame_free(JMDecodedFrame *frame);

// Close and free the decoder.
void jm_decoder_close(JMDecoderContext *ctx);

#endif /* LIBLDECOD_JM_BRIDGE_H */
