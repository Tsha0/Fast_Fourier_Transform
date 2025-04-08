Python version 3.12.4

Running the program:
python fft.py [-m mode] [-i image]
where:
• mode (optional):
– [1] (Default )Fast mode: Convert image to FFT form and display.
– [2] Denoise: The image is denoised by applying an FFT, truncating
high frequencies and then displayed
– [3] Compress: Compress image and plot.
– [4] Plot runtime graphs for the report.
• image (optional): Filename of the image for the DFT (default: given
image).