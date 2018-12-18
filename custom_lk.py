# -*- coding: utf-8 -*-

""" Code developed during 2018 MLH hackathon at the University of Manitoba.

Code adapted from my own old code.

custom_lk.py: this contains the code for implementing the hierarchical Lucade-Kanade
algorithm for optical flow estimation.
"""

import numpy as np
import cv2
import os


class CustomLK:

    def __init__(self, config=None):

        if config is not None:
            self.config = config

    def draw_vectors(self, u, v, scale, stride, color=(0, 255, 0), history=None):
        """ Method for displaying motion vectors on image frame 
        
        Args:
            u (numpy.array): horizontal dim flow vectors
            v (numpy.array): vertical dim flow vectors
            scale (int): rescale size for motion vectors.
            stride (float): increment amount for choosing which motion vectors to
                            display.
            color (tuple): color specified for motion vectors.
            history (numpy.array): array containing history of flow vector counts.

        Returns:
            img_out (numpy.array): image with motion vectors displayed on top.
        """

        img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

        for y in range(0, v.shape[0], stride):

            for x in range(0, u.shape[1], stride):

                if history is not None:
                    hist_val = history[y // stride, x // stride]
                    if hist_val > 1:
                        color = (0, 0, 255)
                    if 0 < hist_val < 4:
                        color = (0, 220, 30)
                    elif 4 <= hist_val < 8:
                        color = (0, 160, 90)
                    elif 8 <= hist_val < 12:
                        color = (0, 120, 120)
                    elif 12 <= hist_val < 16:
                        color = (0, 40, 220)
                    elif hist_val > 16:
                        color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), color, 2)
                cv2.circle(img_out, (x + int(u[y, x] * scale),
                                     y + int(v[y, x] * scale)), 1, color, 1)
        return img_out

    def optic_flow_lk(self, img_a, img_b, k_size, sigma=1):
        """ Computes optic flow using the Lucas-Kanade method.

        Args:
            img_a (numpy.array): grayscale floating-point image with
                                 values in [0.0, 1.0].
            img_b (numpy.array): grayscale floating-point image with
                                 values in [0.0, 1.0].
            k_size (int): size of averaging kernel to use for weighted
                          averages.
            sigma (float): sigma value if gaussian is chosen.

        Returns:
            tuple: 2-element tuple containing:
                U (numpy.array): raw displacement (in pixels) along
                                 X-axis, same size as the input images,
                                 floating-point type.
                V (numpy.array): raw displacement (in pixels) along
                                 Y-axis, same size and type as U.
        """

        # create kernel for weighted averaging
        window = np.ones((k_size, k_size)) / k_size ** 2

        # set tolerance for finding singularities
        tol = 1E-12

        # determine Ix and Iy for either image (here using img_a)
        i_x = cv2.Sobel(img_a, ddepth=-1, dx=1, dy=0, ksize=3, scale=0.125)
        i_y = cv2.Sobel(img_a, ddepth=-1, dx=0, dy=1, ksize=3, scale=0.125)

        # calculate the temporal derivative
        i_t = img_b - img_a

        # calculate IxIx, IxIy and IyIy
        i_xx = np.multiply(i_x, i_x)
        i_xy = np.multiply(i_x, i_y)
        i_yy = np.multiply(i_y, i_y)

        # calculate and compose B in A^T A * d = B
        i_xt = np.multiply(-i_x, i_t)
        i_yt = np.multiply(-i_y, i_t)

        # calculate and compose A^T A using convolution to sum gradients
        sum_i_x = cv2.filter2D(i_xx, ddepth=-1, kernel=window)
        sum_i_xy = cv2.filter2D(i_xy, ddepth=-1, kernel=window)
        sum_i_y = cv2.filter2D(i_yy, ddepth=-1, kernel=window)

        # calculate and compose A^T B to be used in solving for [u v]
        sum_xt = cv2.filter2D(i_xt, ddepth=-1, kernel=window)
        sum_yt = cv2.filter2D(i_yt, ddepth=-1, kernel=window)

        # calculate the determinant of A^T A and zero out any singular points
        det_denom = (np.multiply(sum_i_x, sum_i_y) - np.multiply(sum_i_xy,
                                                                  sum_i_xy)).astype(np.float)
        det_denom[abs(det_denom - 0) < tol] = float('inf')
        det = 1 / det_denom

        # calculate [u, v] = (A^T A)-1 * B
        u = np.multiply(np.multiply(sum_i_y, det), sum_xt)\
            - np.multiply(np.multiply(sum_i_xy, det), sum_yt)
        v = -np.multiply(np.multiply(sum_i_xy, det), sum_xt)\
            + np.multiply(np.multiply(sum_i_x, det), sum_yt)

        return u, v

    def reduce(self, image):
        """ Reduces an image to half its shape (downsampling).

        Args:
            image (numpy.array): grayscale floating-point image, values in
                                 [0.0, 1.0].

        Returns:
            numpy.array: output image with half the shape, same type as the
                         input image.
        """

        # generate 5-tap separable filter and multiply it by itself to get 2d
        # kernel
        oneD_kernel = np.array([1, 4, 6, 4, 1]).reshape(5, 1) / 16.
        generative_kernel = np.dot(oneD_kernel, oneD_kernel.T)

        # filter the image using this 2d generative filter and remove every
        # second row and column from image
        reduced_img = cv2.filter2D(image, ddepth=-1, kernel=generative_kernel)
        reduced_img = reduced_img[::2, ::2]

        return reduced_img

    def gaussian_pyramid(self, image, levels):
        """ Creates a Gaussian pyramid of a given image.

        Args:
            image (numpy.array): grayscale floating-point image, values
                                 in [0.0, 1.0].
            levels (int): number of levels in the resulting pyramid.

        Returns:
            list: Gaussian pyramid, list of numpy.arrays.
        """

        # initialize pyramid as list of images firstly storing original image
        img_list = [np.copy(image)]

        # iterate over number of remaining levels
        for _ in range(levels - 1):

            # run the REDUCE function to correctly downsample the image
            reduced_image = self.reduce(image)

            img_list.append(reduced_image)

            image = reduced_image

        return img_list

    def expand(self, image):
        """ Expands an image doubling its width and height (upsampling).

        Args:
            image (numpy.array): grayscale floating-point image, values
                                 in [0.0, 1.0].

        Returns:
            numpy.array: same type as 'image' with the doubled height and
                         width.
        """

        # generate 5-tap separable filter and multiply it by itself to get 2d
        # kernel
        oneD_kernel = np.array([1, 4, 6, 4, 1]).reshape(5, 1) / 16.
        generative_kernel = np.dot(oneD_kernel, oneD_kernel.T) * 4.

        # filter the image using this 2d generative filter and remove every
        # second row and column from image
        expanded_img = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
        expanded_img[::2, ::2] = image
        expanded_img = cv2.filter2D(expanded_img, ddepth=-1,
                                   kernel=generative_kernel)

        return expanded_img

    def laplacian_pyramid(self, g_pyr):
        """ Creates a Laplacian pyramid from a given Gaussian pyramid.

        Args:
            g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

        Returns:
            list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
        """

        # initialize l_pyr as empty list
        l_pyr = []

        # store the larger non-expanded image
        higher_image = g_pyr[0]

        # iterate over remaining images to create difference images for laplacian
        for level in g_pyr[1:]:

            # expand next level by upsampling
            expanded_image = self.expand(level)

            # if higher image is an odd shape then remove final row/col
            if higher_image.shape[0] % 2 != 0:
                expanded_image = expanded_image[:-1, :]
            if higher_image.shape[1] % 2 != 0:
                expanded_image = expanded_image[:, :-1]

            # get difference image and add to laplacian pyramid
            image_diff = higher_image - expanded_image
            l_pyr.append(image_diff)

            higher_image = level

        # fill last level of laplacian pyramid with last gaussian image
        l_pyr.append(g_pyr[-1])

        return l_pyr

    def warp(self, image, U, V, interpolation, border_mode):
        """ Warps image using X and Y displacements (U and V).

        Args:
            image (numpy.array): grayscale floating-point image, values
                                 in [0.0, 1.0].
            U (numpy.array): displacement (in pixels) along X-axis.
            V (numpy.array): displacement (in pixels) along Y-axis.
            interpolation (Inter): interpolation method used in cv2.remap.
            border_mode (BorderType): pixel extrapolation method used in
                                      cv2.remap.

        Returns:
            numpy.array: warped image, such that
                         warped[y, x] = image[y + V[y, x], x + U[y, x]]
        """

        # create meshgrid storing coordinates for x's and y's in image
        x_mesh, y_mesh = np.meshgrid(range(image.shape[1]),
                                     range(image.shape[0]))

        # check if there are extra rows or cols in U and V
        extra_x = U.shape[0] - image.shape[0]
        extra_y = U.shape[1] - image.shape[1]

        # trim U and V if necessary
        if extra_x:
            U = U[: -extra_x, :]
            V = V[: -extra_x, :]

        if extra_y:
            U = U[:, : -extra_y]
            V = V[:, : -extra_y]

        # add displacements u and v to meshgrid to get coordinate changes
        u_map = (x_mesh + U).astype(np.float32)
        v_map = (y_mesh + V).astype(np.float32)

        # remap image using coordinate changes from u_map and v_map
        remapped_img = cv2.remap(image.astype(np.float32), map1=u_map,
                                 map2=v_map,
                                 interpolation=interpolation,
                                 borderMode=border_mode)

        return remapped_img

    def hierarchical_lk(self, img_a, img_b, orig_b, levels, k_size,
                        sigma, interpolation, border_mode, mask=None,
                        history=None, stride=10):
        """ Computes the optic flow using Hierarchical Lucas-Kanade.

        Args:
            img_a (numpy.array): grayscale floating-point image, values in
                                 [0.0, 1.0].
            img_b (numpy.array): grayscale floating-point image, values in
                                 [0.0, 1.0].
            levels (int): Number of levels.
            k_size (int): parameter to be passed to optic_flow_lk.
            sigma (float): parameter to be passed to optic_flow_lk.
            interpolation (Inter): parameter to be passed to warp.
            border_mode (BorderType): parameter to be passed to warp.

        Returns:
            tuple: 4-element tuple containing:
                U (numpy.array): raw displacement (in pixels) along X-axis,
                                 same size as the input images as float.
                V (numpy.array): raw displacement (in pixels) along Y-axis,
                                 same size as the input images as float.
                img (numpy.array): image of passed in frame along with motion
                                   vectors displayed.
                img_b (numpy.array): image of original passed in secondary frame.
                history (numpy.array): history of flow vector counts.
        """

        # create the gaussian pyramids for each image
        pyr_a = self.gaussian_pyramid(img_a, levels)
        pyr_b = self.gaussian_pyramid(img_b, levels)

        # get the final images in each pyramid as a starting point
        final_a = pyr_a[-1]
        final_b = pyr_b[-1]

        # determine initial flow fields u and v
        u, v = self.optic_flow_lk(final_a, final_b,
                             k_size, sigma)

        # expand u and v and double values
        u = self.expand(u)
        v = self.expand(v)

        u, v = u * 2, v * 2

        for level in range(levels - 2, -1, -1):
            # get the next two images in the gaussian pyramids
            img_a_reduced = pyr_a[level]
            img_b_reduced = pyr_b[level]

            # warp b towards a using existing u and v
            warped_img_b = self.warp(img_b_reduced, u, v,
                                interpolation=interpolation,
                                border_mode=border_mode)

            # get the delta_u and delta_v based on flow between warped b and
            # current a image
            delta_u, delta_v = self.optic_flow_lk(img_a_reduced, warped_img_b,
                                 k_size, sigma)

            # as before check to make there aren't extra rows/cols due to expand
            # operation
            extra_x = u.shape[0] - delta_u.shape[0]
            extra_y = u.shape[1] - delta_u.shape[1]

            # trim if there are extra rows/cols
            if extra_x:
                u = u[: -extra_x, :]
                v = v[: -extra_x, :]

            if extra_y:
                u = u[:, : -extra_y]
                v = v[:, : -extra_y]

            # update flow fields
            u += delta_u
            v += delta_v

            # if haven't reached base level then expand and double to continue alg
            if level > 0:
                u = self.expand(u)
                v = self.expand(v)

                u, v = u * 2, v * 2

        u = u / np.max(u)
        v = v / np.max(v)

        u[:, :400] *= 0.4
        v[:, :400] *= 0.4

        if mask is not None:
            u *= mask
            v *= mask

        # draw motion vectors on image prior to returning resulting image
        img = self.draw_vectors(u, v, scale=75, stride=stride, history=history)
        if orig_b.shape[:2] != (640, 1140):
            orig_b = orig_b[40: 680, 70: 1210]
        img = cv2.add(orig_b, img)

        return u, v, img, img_b, history
