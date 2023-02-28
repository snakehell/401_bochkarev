using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
namespace ONNX.FastNeuralStyleTransfer;


public class StyleTransfer
{
    private InferenceSession? session = null;
    static Semaphore sem = new Semaphore(1, 1);
    const int Width = 224;
    const int Height = 224;
    const int Channels = 3;
    int i = 0;
    public async Task<string> Process(string path, CancellationToken token)
    {

        Image<Rgb24> original = Image.Load<Rgb24>(path);
        return await Task<string>.Factory.StartNew(() => {
            var image = original.Clone(ctx =>
            {
                ctx.Resize(new ResizeOptions
                {
                    Size = new Size(Width, Height),
                    Mode = ResizeMode.Crop
                });
            });

            using var session = new InferenceSession($@"./model/mosaic-9.onnx");
            sem.WaitOne();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results
            = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("input1", ImageToTensor(image))
            });
            sem.Release();
            if (results.FirstOrDefault()?.Value is not Tensor<float> output)
                throw new ApplicationException("Unable to process image");
            IImageEncoder encoder = Path.GetExtension(path) switch
            {
                ".png" => new PngEncoder(),
                ".jpg" => new JpegEncoder(),
                _ => throw new Exception()
            };
            string path_mosaic = $@"C:\Users\Семён\Desktop\StyleTransfer-main\src\output/{i}_mosaic.png";
            i++;
            TensorToImage(output).Save(path_mosaic, encoder);
            return path_mosaic;
        }, token, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    }
    
    Image<Rgb24> TensorToImage(Tensor<float>output) {
        var result = new Image<Rgb24>(Width, Height);
        for (var y = 0; y < Height; y++)
        {
            for (var x = 0; x < Width; x++)
            {
                result[x, y] = new Rgb24(
                    FloatPixelValueToByte(output[0, 0, y, x]),
                    FloatPixelValueToByte(output[0, 1, y, x]),
                    FloatPixelValueToByte(output[0, 2, y, x])
                );
            }
        }
        return result;
    }
    DenseTensor<float> ImageToTensor(Image<Rgb24> img) {
        var input = new DenseTensor<float>(new[] { 1, Channels, Height, Width });
        for (var y = 0; y < img.Height; y++)
        {
            var pixelSpan = img.GetPixelRowSpan(y);
            for (var x = 0; x < img.Width; x++)
            {
                input[0, 0, y, x] = pixelSpan[x].R;
                input[0, 1, y, x] = pixelSpan[x].G;
                input[0, 2, y, x] = pixelSpan[x].B;
            }
        }
        return input;
    }
    static byte FloatPixelValueToByte(float pixelValue) =>
        (byte) Math.Clamp(pixelValue, 0, 255);
}