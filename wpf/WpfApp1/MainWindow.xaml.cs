using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;
using System.Collections.ObjectModel;
using System.Windows.Markup;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using ONNX.FastNeuralStyleTransfer;

namespace WpfApp
{
    
        public class Image
        {

            private string path;
            private string path_mosaic;
            public string Path
            {
                get { return this.path; }
                set { this.path = value; }
            }
        public string PathMosaic
        {
            get { return this.path_mosaic; }
            set { this.path_mosaic = value; }
        }
        public Image(string path, string path_mosaic)
            {
                this.path = path;
                this.path_mosaic = path_mosaic;
        }
        }

        public partial class MainWindow : Window
        {
            StyleTransfer st;
            CancellationToken token;
            CancellationTokenSource source;
            ObservableCollection<Image> img = new();
            ObservableCollection<Image> img_mosaic = new();
            public MainWindow()
            {
                InitializeComponent();
                this.st = new StyleTransfer();
                lv_neutral.ItemsSource = img;
            }

            private async void command_open(object sender, RoutedEventArgs e)
            {
                Files_selection.IsEnabled = false;
                pbStatus.Value = 0;
                this.source = new CancellationTokenSource();
                this.token = source.Token;
                Microsoft.Win32.OpenFileDialog dialog = new Microsoft.Win32.OpenFileDialog();
                dialog.Multiselect = true;
                dialog.Filter = "(*.jpg, *.png)|*.jpg; *.png";
                var result = dialog.ShowDialog();
                int len = dialog.FileNames.Length;
                if (len == 0)
                {
                    Files_selection.IsEnabled = true;
                    return;
                }
                int step = 500 / len;
                var tasks = new List<Task<string>>();
                var paths = dialog.FileNames;
                if (result == true)
                {
                    for (int i = 0; i < len; i++)
                    {
                        var path = dialog.FileNames[i];
                        var tmp = st.Process(path, token);
                        tasks.Add(tmp);
                        pbStatus.Value += step;
                    }
                }
                bool flag = true;
                for (int i = 0; i < len; i++)
                {
                    await tasks[i];
                    string res = tasks[i].Result;
                    if (res == null)
                    {
                        flag = false;
                        break;
                    }
                    img.Add(new Image(paths[i], res));   
                    pbStatus.Value += step;
                }
                if (flag)
                    pbStatus.Value = 1000;
                Files_selection.IsEnabled = true;
            }

            private void command_stop(object sender, RoutedEventArgs e)
            {
                source.Cancel();
                MessageBox.Show("Cancelled");
            }

        }
    
}
