void _gs_render_batch(
        const float *sigmas,
        const float *coords,
        const float *colors,
        float *rendered_img,
        const int b,
	const int s, 
	const int h, 
	const int w, 
	const int c,
	const float dmax
);

void _gs_render_backward_batch(
        const float *sigmas,
        const float *coords,
        const float *colors,
        const float *grads,
        float *grads_sigmas,
        float *grads_coords,
        float *grads_colors,
        const int b,
	const int s, 
	const int h, 
	const int w, 
	const int c,
	const float dmax
);
