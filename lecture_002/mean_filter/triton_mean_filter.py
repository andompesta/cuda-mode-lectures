# import os
# os.environ["TRITON_INTERPRET"] = "1"

import torch
import triton
import triton.language as tl
from torchvision.io import read_image, write_png



@triton.jit
def kernel_mean_filter(
    output_ptr,
    input_ptr,
    width,
    height,
    stride_x,
    stride_y,
    stride_z,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    RADIUS: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of output it should compute.
    pid_c = tl.program_id(0)  # row
    pid_y = tl.program_id(1)  # col
    pid_x = tl.program_id(2)  # channel

    rows = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)  # shape (BLOCK_M,)
    cols = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # shape (BLOCK_N,)
    channel = pid_c

    # tmp values
    pixval = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.int32)
    pixels = tl.zeros([BLOCK_SIZE_Y, BLOCK_SIZE_X], dtype=tl.int32)
    # create all deltas in 2D
    dy = tl.arange(0, 2 * RADIUS)[:, None] - RADIUS # shape (RADIUS, 1)
    dx = tl.arange(0, 2 * RADIUS)[None, :] - RADIUS  # shape (1, RADIUS)
    # broadcast to 4D: height, width, dy, dx
    cur_y = rows[:, None, None, None] + dy[None, None, :, :]  # shape (BLOCK_Y, 1, RADIUS, 1)
    cur_x = cols[None, :, None, None] + dx[None, None, :, :]  # shape (1, BLOCK_X, 1, RADIUS)

    mask = (cur_y >= 0) & (cur_y < height) & (cur_x >= 0) & (cur_x < width)
    offs = channel * stride_z + cur_y * stride_y + cur_x * stride_x

    vals = tl.load(input_ptr + offs, mask=mask, other=0).to(tl.int32)
    pixval = vals.sum(axis=-1).sum(axis=-1)
    pixels = mask.to(tl.int32).sum(axis=-1).sum(axis=-1)

    # for dy in range(-8, 8 + 1):
    #     for dx in range(-8, 8 + 1):
    #         cur_y = row_ids + dy
    #         cur_c = col_ids + dx
    #         mask = (cur_y >= 0) & (cur_y < height) & (cur_c >= 0) & (cur_c < width)

    #         offs = (channel * stride_z) + (cur_y * stride_y) + (cur_c * stride_x)
    #         vals = tl.load(input_ptr + offs, mask=mask, other=0).to(tl.int32)
    #         pixval += vals
    #         pixels += mask.to(tl.int32)
    mean = pixval // pixels  # (BLOCK_Y, BLOCK_X)

    # get offsets and mask for output
    row_ids = rows[:, None]
    col_ids = cols[None, :]
    offs_out = (channel * stride_z) + (row_ids * stride_y) + (col_ids * stride_x)
    mask_rc = (row_ids < height) & (col_ids < width)

    tl.store(
        output_ptr + offs_out,
        mean.to(tl.uint8),
        mask=mask_rc,
    )


def mean_filter(
    x: torch.Tensor,
    radius: int,
):
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    assert x.dtype == torch.uint8
    C, H, W = x.shape
    output = torch.empty_like(x)

    grid = (
        C,
        (H + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y,
        (W + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X,
    )

    kernel_mean_filter[grid](
        output,
        x,
        W,
        H,
        x.stride(-1),
        x.stride(-2),
        x.stride(-3),
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
        radius,
    )
    return output


def main():
    """
    Use torch cpp inline extension function to compile the kernel in mean_filter_kernel.cu.
    Read input image, convert apply mean filter custom cuda kernel and write result out into output.png.
    """
    # ext = compile_extension()

    x = read_image("lecture_002/mean_filter/Grace_Hopper.jpg").contiguous().cuda()
    print("Input image:", x.shape, x.dtype)

    y = mean_filter(x, 8)

    print("Output image:", y.shape, y.dtype)
    write_png(y.cpu(), "triton_output.png")


if __name__ == "__main__":
    main()
