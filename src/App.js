
import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';
import * as ort from 'onnxruntime-web';

import Nav from './Nav';

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

const sleep = (i = 1) => new Promise(s => {
  const frame = () =>
    window.requestAnimationFrame(--i > 0? frame: s);
  frame();
});
const time2str = t => {
  return t > 60000
    ? ~~(t / 6000) / 10 + 'min'
    : t > 1000
    ? ~~(t / 100) / 10 + 'sec'
    : ~~(t * 10) / 10 + 'ms';
};

function App() {
  const canvasRef = useRef(null), imgRef = useRef(null), maskCanvasRef = useRef(null);
  const [showMask, setShowMask] = useState(false);
  const [isInferencing, setIsInferencing] = useState(false);
  const [showDragCover, setShowDragCover] = useState(false);
  const emptyTimeRecord = () => Object.fromEntries(
    'read,onnx,zoom,convert,inferencing,invNorm,result,mask,total'
    .split(',').map(k => [k, { t: 0, s: '' }])
  );
  const [timeRecord, setTimeRecord] = useState(emptyTimeRecord());
  const handleClick = useCallback(async () => {
    if (!canvasRef.current || !imgRef) return;

    setIsInferencing(true);
    await sleep(2);

    let tr = emptyTimeRecord();
    setTimeRecord(tr);
    const setRecord = (k, dt) => setTimeRecord(tr = { ...tr,
      [k]: { t: dt, s: time2str(dt) }
    }) || sleep();

    const zoomHelperCanvas = document.createElement('canvas'), zoomHelperCtx = zoomHelperCanvas.getContext('2d', { willReadFrequently: true });
    const zoomImg = (img, w = 320, h = 320, canvas = zoomHelperCanvas, ctx = zoomHelperCtx) => {
      canvas.width = w;
      canvas.height = h;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, w, h);
      return ctx.getImageData(0, 0, w, h);
    };
    const canvas = canvasRef.current, ctx = canvas.getContext('2d', { willReadFrequently: true }), img = imgRef.current;

    console.log('running');
    let t = performance.now(), lt = t;
    const recording = k =>
      setRecord(k, performance.now() - lt)
      .then(() => lt = performance.now());

    const inputImgData = zoomImg(img, img.naturalWidth, img.naturalHeight, canvas, ctx);
    await recording('read');

    let session = await inferenceSession;
    await recording('onnx');

    // zoom the image to 320 x 320
    // The original uses a Gaussian filter for smoothing, but here
    // the browser's built-in smooth zoom is used instead of skimage.transform.resize(anti_aliasing=True).
    // Original: https://github.com/xuebinqin/U-2-Net/blob/53dc9da026650663fc8d8043f3681de76e91cfde/data_loader.py#L40
    // Gaussian: https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=transform#skimage.transform.resize
    const zoomedInputImgData = zoomImg(img, 320, 320, canvas, ctx),
      len = zoomedInputImgData.data.length,
      w = 320, h = 320;
    await recording('zoom');

    // Normalize the input, and convert it to tensor.
    const inputTensor = new Float32Array(len / 4 * 3);
    const [MEAN_R, MEAN_G, MEAN_B] = [.485, .456, .406];
    const [STD_R, STD_G, STD_B] = [.229, .224, .225];
    for (let i = 0, j = 0, sr = 0, sg = len >> 2, sb = len >> 1; i < len; i = (++j) << 2) {
      inputTensor[sr + j] = (zoomedInputImgData.data[i + 0] / 255 - MEAN_R) / STD_R;
      inputTensor[sg + j] = (zoomedInputImgData.data[i + 1] / 255 - MEAN_G) / STD_G;
      inputTensor[sb + j] = (zoomedInputImgData.data[i + 2] / 255 - MEAN_B) / STD_B;
    }
    await recording('convert');
    const output = await session.run({
      'input.1': new ort.Tensor('float32', inputTensor, [1, 3, h, w])
    });
    await recording('inferencing');
    const pred = normPRED(output['1959'].data, 255);
    await recording('invNorm');

    // Put the result to alpha channal.
    for (let i = 0; i < len; i += 4)
      zoomedInputImgData.data[i + 3] = pred[i >> 2];
    zoomHelperCanvas.width = zoomHelperCanvas.height = 320;
    zoomHelperCtx.putImageData(zoomedInputImgData, 0, 0, 0, 0, w, h);
    const outputImgData = zoomImg(zoomHelperCanvas, img.naturalWidth, img.naturalHeight, canvas, ctx);
    for (let i = 0; i < inputImgData.data.length; i += 4) {
      inputImgData.data[i + 3] = outputImgData.data[i + 3];
    }
    ctx.putImageData(inputImgData, 0, 0);
    await recording('result');

    const maskCanvas = maskCanvasRef.current, maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
    maskCanvas.width = img.naturalWidth;
    maskCanvas.height = img.naturalHeight;
    for (let i = 0; i < outputImgData.data.length; i += 4) {
      outputImgData.data[i + 0] = outputImgData.data[i + 1] = outputImgData.data[i + 2] = outputImgData.data[i + 3];
      outputImgData.data[i + 3] = 255;
    }
    maskCtx.putImageData(outputImgData, 0, 0);
    await recording('mask');

    await setRecord('total', performance.now() - t);
    await (new Promise(s => setTimeout(s, 2000)));
    setIsInferencing(false);
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
  const handleDragEnter = useCallback(e => {
    e.preventDefault();
    setShowDragCover(true);
  }, []);
  const handleDragLeave = useCallback(e => {
    if (e.target.id !== 'react-app-container' && !e.target.className.includes('drag-cover'))
      return;
    setShowDragCover(false);
  }, []);
  const handleDrop = useCallback(e => {
    e.preventDefault();
    setShowDragCover(false);
    handleFileSelect({target: e.dataTransfer});
  }, [handleFileSelect]);
  const handlePaste = useCallback(e => {
    handleFileSelect({target: e.clipboardData});
  }, [handleFileSelect]);
  const handleDownload = useCallback(e => {
    const mine = 'image/png';
    const a = document.getElementById('download_a');
    const canvas = (showMask? maskCanvasRef: canvasRef).current;
    a.href = canvas.toDataURL(mine, 1).replace(mine, 'image/octet-stream');
    a.click();
  }, [showMask]);
  const toggleMaskBtnClick = useCallback(e => {
    setShowMask(!showMask);
  }, [showMask]);
  return <div id='react-app-container'
      onPaste={handlePaste}
      onDragEnter={handleDragEnter}
      onDrop={handleDrop}
      onDragOver={handleDragEnter}
      onDragLeaveCapture={handleDragLeave}
      onDragExitCapture={handleDragLeave}
  >
    <Nav/>
    <main>
      <section id='section-kanban'>
        <div id='kanban-musume'>
          <img alt='kanban musume transparent' className='stripe-bg' src='./kanban_musume_t.png' />
          <img alt='kanban musume' src='./kanban_musume.jpg' />
        </div>
      </section>
      <section id='section-about'>
        <h2>About</h2>
        This project attempts to run <a href='https://github.com/xuebinqin/U-2-Net'>U<sup>2</sup>Net</a> in a web browser.
        <br />
        The project is based on the reduced-size U<sup>2</sup>Net model proposed in <a href='https://github.com/xuebinqin/U-2-Net/issues/295'>silueta.me</a>,
        <br />
        and is inferencing in the browser through the ONNX web runtime.
        <br />
        See <a href='https://github.com/jerrychan7/U2netInBrowser'>README.md</a> for more details.
      </section>
      <section id='section-function'>
        <div className='toolbar'>
          <span>DROP it anywhere</span>
          <label className='file-btn btn' htmlFor='file-selector'>select your image</label>
          <span>Ctrl + v to PASTE it</span>
        </div>
        <div className='echo-area'>
          <img ref={imgRef} alt='input' id='input' className='stripe-bg' src='./kanban_musume.jpg' />
          <div className='btn inferencing' onClick={handleClick}>click</div>
          <div className='output-wrap'>
            <canvas ref={canvasRef} id='output' className='stripe-bg' />
            <canvas ref={maskCanvasRef} id='mask-output' style={{ display: showMask? '': 'none' }} />
            <div className='time-board' style={{ display: isInferencing? '': 'none' }}>
              <table>
                <thead><tr><th>Magic</th>       <th>Time</th></tr></thead>
                <tbody>
                  <tr><td>Read Image:</td> <td>{timeRecord.read.s}</td></tr>
                  <tr><td>ONNX Ready:</td> <td>{timeRecord.onnx.s}</td></tr>
                  <tr><td>Zoom Image:</td> <td>{timeRecord.zoom.s}</td></tr>
                  <tr><td>Convert:</td>    <td>{timeRecord.convert.s}</td></tr>
                  <tr><td>Inferencing:</td><td>{timeRecord.inferencing.s}</td></tr>
                  <tr><td>InvNorm:</td>    <td>{timeRecord.invNorm.s}</td></tr>
                  <tr><td>Gen Result:</td> <td>{timeRecord.result.s}</td></tr>
                  <tr><td>Gen Mask:</td>   <td>{timeRecord.mask.s}</td></tr>
                </tbody>
                <tfoot><tr><th>Total:</th> <td>{timeRecord.total.s}</td></tr></tfoot>
              </table>
            </div>
          </div>
        </div>
        <div className='toolbar2'>
          <button className={showMask? 'btn': 'btn stripe-bg'} onClick={toggleMaskBtnClick}>mask</button>
          <button className='btn' onClick={handleDownload}>download</button>
        </div>
        <p>
          The app can only handle images. If it's a .gif, only the first frame can be processed.
        </p>
      </section>
    </main>
    <input id='file-selector' type='file' onChange={handleFileSelect} hidden />
    <div className={'drag-cover' + (showDragCover? ' show': '')}
      onDragLeaveCapture={handleDragLeave}
      onDragExitCapture={handleDragLeave}
    >
      <p>Drop your image file here to find the salient object!</p>
    </div>
    <a id='download_a' download='result.png'></a>
  </div>;
}

export default App;
