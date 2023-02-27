
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

// const sleep0 = () => new Promise(s => window.requestAnimationFrame(s));
// const time2str = t => {
//   return t > 60000
//     ? ~~(t / 6000) / 10 + 'min'
//     : t > 1000
//     ? ~~(t / 100) / 10 + 'sec'
//     : ~~(t * 10) / 10 + 'ms';
// };

function App() {
  const canvasRef = useRef(null), imgRef = useRef(null);
  // const [durationTime, setDurationTime] = useState(0);
  // const [showDropCover, setShowDropCover] = useState(false);
  // const [timeRecord, setTimeRecord] = useState({
  //   'read': { t: 0, s: '' },
  //   'onnx': { t: 0, s: '' },
  //   'zoom': { t: 0, s: '' },
  //   'convert': { t: 0, s: '' },
  //   'inferencing': { t: 0, s: '' },
  //   'invNorm': { t: 0, s: '' },
  //   'mask': { t: 0, s: '' },
  //   'result': { t: 0, s: '' },
  //   'total': { t: 0, s: '' },
  // });
  const handleClick = useCallback(async () => {
    if (!canvasRef.current || !imgRef) return;

    // setTimeRecord(Object.fromEntries(Object.keys(timeRecord).map(k => [k, { t: 0, s: '...' }])));
    // const setRecord = (k, dt) => setTimeRecord({ ...timeRecord,
    //   [k]: { t: dt, s: time2str(dt) }
    // }) || sleep0();

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
    // const recording = k =>{ console.log(JSON.stringify(timeRecord), k, performance.now() - lt, time2str(performance.now() - lt));
    //   return setRecord(k, performance.now() - lt)
    //   .then(() => lt = performance.now());}

    // const inputImgData = zoomImg(img, img.naturalWidth, img.naturalHeight, canvas, ctx);
    // await recording('read');

    let session = await inferenceSession;
    // await recording('onnx');

    // zoom the image to 320 x 320
    // The original uses a Gaussian filter for smoothing, but here
    // the browser's built-in smooth zoom is used instead of skimage.transform.resize(anti_aliasing=True).
    // Original: https://github.com/xuebinqin/U-2-Net/blob/53dc9da026650663fc8d8043f3681de76e91cfde/data_loader.py#L40
    // Gaussian: https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=transform#skimage.transform.resize
    const zoomedInputImgData = zoomImg(img),
      len = zoomedInputImgData.data.length,
      w = 320, h = 320;
    // await recording('zoom');

    // Normalize the input, and convert it to tensor.
    const inputTensor = new Float32Array(len / 4 * 3);
    const [MEAN_R, MEAN_G, MEAN_B] = [.485, .456, .406];
    const [STD_R, STD_G, STD_B] = [.229, .224, .225];
    for (let i = 0, j = 0, sr = 0, sg = len >> 2, sb = len >> 1; i < len; i = (++j) << 2) {
      inputTensor[sr + j] = (zoomedInputImgData.data[i + 0] / 255 - MEAN_R) / STD_R;
      inputTensor[sg + j] = (zoomedInputImgData.data[i + 1] / 255 - MEAN_G) / STD_G;
      inputTensor[sb + j] = (zoomedInputImgData.data[i + 2] / 255 - MEAN_B) / STD_B;
    }
    // await recording('convert');
    const output = await session.run({
      'input.1': new ort.Tensor('float32', inputTensor, [1, 3, h, w])
    });
    // await recording('inferencing');
    const pred = normPRED(output['1959'].data, 255);
    // await recording('invNorm');

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

    // setDurationTime(performance.now() - t);
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
    console.log('DragEnter', e);
    // setShowDropCover(true);
  }, []);
  const handleDragLeave = useCallback(e => {
    console.log('DragLeave', e);
    // setShowDropCover(false);
  }, []);
  const handleDrop = useCallback(e => {
    e.preventDefault();
    handleFileSelect({target: e.dataTransfer});
  }, [handleFileSelect]);
  const handlePaste = useCallback(e => {
    handleFileSelect({target: e.clipboardData});
  }, [handleFileSelect]);
  const handleDownload = useCallback(e => {
    const mine = 'image/png';
    const a = document.getElementById('download_a');
    a.href = canvasRef.current.toDataURL(mine, 1).replace(mine, 'image/octet-stream');
    a.click();
  }, []);
  const toggleMaskBtnClick = useCallback(e => {}, []);
  return <div id='react-app-container' onDragEnter={handleDragEnter} onDragOver={e => e.preventDefault()} onDrop={handleDrop} onPaste={handlePaste} onDragLeave={handleDragLeave}>
    <Nav />
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
            {/* <div className='time-board'>
              <table>
                <thead><tr><th></th>       <th>Time</th></tr></thead>
                <tbody>
                  <tr><td>Read Image:</td> <td>{timeRecord.read.s}</td></tr>
                  <tr><td>ONNX Ready:</td> <td>{timeRecord.onnx.s}</td></tr>
                  <tr><td>Zoom Image:</td> <td>{timeRecord.zoom.s}</td></tr>
                  <tr><td>Convert:</td>    <td>{timeRecord.convert.s}</td></tr>
                  <tr><td>Inferencing:</td><td>{timeRecord.inferencing.s}</td></tr>
                  <tr><td>InvNorm:</td>    <td>{timeRecord.invNorm.s}</td></tr>
                  <tr><td>Gen Mask:</td>   <td>{timeRecord.mask.s}</td></tr>
                  <tr><td>Gen Result:</td> <td>{timeRecord.result.s}</td></tr>
                </tbody>
                <tfoot><tr><th>Total:</th> <td>{timeRecord.total.s}</td></tr></tfoot>
              </table>
            </div> */}
          </div>
        </div>
        <div className='toolbar2'>
          {/* <button className='btn' onClick={toggleMaskBtnClick}>mask</button> */}
          <button className='btn' onClick={handleDownload}>download</button>
        </div>
        <p>
          {/* <span>Total: {~~(durationTime / 10) / 100} second</span> */}
          The app can only handle images. If it's a gif, only the first frame can be processed.
        </p>
      </section>
    </main>
    <input id='file-selector' type='file' onChange={handleFileSelect} hidden />
    {/* <div className='drop-cover' style={{ display: showDropCover? '': 'none' }}>
      <p>Drop your image file here to find the salient object!</p>
    </div> */}
    <a id='download_a' download='result.png'></a>
  </div>;
}

export default App;
