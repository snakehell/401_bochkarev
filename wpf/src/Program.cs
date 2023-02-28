using ONNX.FastNeuralStyleTransfer;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;



Directory.CreateDirectory("output");
StyleTransfer st = new StyleTransfer();



var imageFolderPath = @$"C:\Users\Семён\Desktop\StyleTransfer-main\src\input";
string[] allfiles = Directory.GetFiles(imageFolderPath);

foreach (string filename in allfiles)
{
    string imageFilePath = @$"{filename}";
    CancellationTokenSource source = new CancellationTokenSource();
    CancellationToken token = source.Token;
    var result = st.Process(imageFilePath, token);
    await result;

}







