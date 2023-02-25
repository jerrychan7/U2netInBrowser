
import { useCallback, useRef, useState } from 'react';
import './App.css';
import * as ort from 'onnxruntime-web';

// ort.env.wasm.numThreads = 8;
const inferenceSession = ort.InferenceSession.create('./u2net.quant.onnx', {
  // onnx web runtime with webgl not supporting 'DynamicQuantizeLinear' operator yet
  // https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/operators.md
  // executionProviders: ['webgl'],
  executionProviders: ['wasm'],
  // executionMode: 'parallel',
  // enableCpuMemArena: true,
});

const normPRED = (d, s = 1) => {
  let ma = Number.MIN_VALUE, mi = Number.MAX_VALUE;
  for (let i of d) {
    if (ma < i) ma = i;
    if (mi > i) mi = i;
  }
  return d.map(i => ((i - mi) / (ma - mi)) * s);
};

function App() {
  const canvasRef = useRef(null), imgRef = useRef(null);
  const [durationTime, setDurationTime] = useState(0);
  const handleClick = useCallback(async () => {
    if (!canvasRef.current || !imgRef) return;

    const zoomHelperCanvas = document.createElement('canvas'), zoomHelperCtx = zoomHelperCanvas.getContext('2d');
    const zoomImg = (img, w = 320, h = 320, canvas = zoomHelperCanvas, ctx = zoomHelperCtx) => {
      canvas.width = w;
      canvas.height = h;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, w, h);
      return ctx.getImageData(0, 0, w, h);
    };
    const canvas = canvasRef.current, ctx = canvas.getContext('2d'), img = imgRef.current;

    console.log('running');
    let t = performance.now();
    let session = await inferenceSession;

    // zoom the image to 320 x 320
    // The original uses a Gaussian filter for smoothing, but here
    // the browser's built-in smooth zoom is used instead of skimage.transform.resize(anti_aliasing=True).
    // Original: https://github.com/xuebinqin/U-2-Net/blob/53dc9da026650663fc8d8043f3681de76e91cfde/data_loader.py#L40
    // Gaussian: https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=transform#skimage.transform.resize
    const zoomedInputImgData = zoomImg(img),
      len = zoomedInputImgData.data.length,
      w = 320, h = 320;

    // Normalize the input, and convert it to tensor.
    const inputTensor = new Float32Array(len / 4 * 3);
    const [MEAN_R, MEAN_G, MEAN_B] = [.485, .456, .406];
    const [STD_R, STD_G, STD_B] = [.229, .224, .225];
    for (let i = 0, j = 0, sr = 0, sg = len >> 2, sb = len >> 1; i < len; i = (++j) << 2) {
      inputTensor[sr + j] = (zoomedInputImgData.data[i + 0] / 255 - MEAN_R) / STD_R;
      inputTensor[sg + j] = (zoomedInputImgData.data[i + 1] / 255 - MEAN_G) / STD_G;
      inputTensor[sb + j] = (zoomedInputImgData.data[i + 2] / 255 - MEAN_B) / STD_B;
    }
    const output = await session.run({
      'input.1': new ort.Tensor('float32', inputTensor, [1, 3, h, w])
    });
    const pred = normPRED(output['1959'].data, 255);

    // Put the result to alpha channal.
    for (let i = 0; i < len; i += 4)
      zoomedInputImgData.data[i + 3] = pred[i >> 2];
    zoomHelperCtx.putImageData(zoomedInputImgData, 0, 0, 0, 0, w, h);
    const outputImgData = zoomImg(zoomHelperCanvas, img.naturalWidth, img.naturalHeight, canvas, ctx);
    const inputImgData = zoomImg(img, img.naturalWidth, img.naturalHeight);
    for (let i = 0; i < inputImgData.data.length; i += 4) {
      outputImgData.data[i + 0] = inputImgData.data[i + 0];
      outputImgData.data[i + 1] = inputImgData.data[i + 1];
      outputImgData.data[i + 2] = inputImgData.data[i + 2];
    }
    ctx.putImageData(outputImgData, 0, 0);

    setDurationTime(performance.now() - t);
  }, []);
  const handleFileSelect = useCallback(({target: { files }}) => {
    for (let i = 0; i < files.length; ++i) {
      const file = files[i];
      if (!(/^image\/.*$/i.test(file.type))) continue;
      const freader = new FileReader();
      freader.onload = e => imgRef.current.src = e.target.result;
      freader.readAsDataURL(file);
      break;
    }
  }, []);
  return (
    <div>
      <p><input type='file' onChange={handleFileSelect} /> <button onClick={handleClick}>run</button> <span>Total: {~~(durationTime / 10) / 100} second</span></p>
      <img ref={imgRef} alt='input' id='input' src='./silueta_me_test.png' />
      <canvas ref={canvasRef} id='output' />
    </div>
  );
}

export default App;
