import * as tf from '@tensorflow/tfjs';
import { Image } from 'react-native';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system'
import labels from "../labels.json"
import { renderBoxes } from './draw';

async function imgToTensor(uri) {

    const img64 = await FileSystem.readAsStringAsync(uri, {
      encoding:FileSystem.EncodingType.Base64
    });
    
    const imgBuffer = tf.util.encodeString(img64, 'base64').buffer
    const raw = new Uint8Array(imgBuffer);
    let imgTensor = decodeJpeg(raw);
  
    return imgTensor;
  }

const preprocess = async (img, modelWidth, modelHeight) => {
// refresh
    let xRatio, yRatio;

    const input = tf.tidy(() => {
    
    const [h, w] = img.shape.slice(0, 2);

    const maxSize = Math.max(w, h);
    const imgPadded = img.pad([
      [0, maxSize - h],
      [0, maxSize - w],
      [0, 0]
    ]);

    xRatio = maxSize / w;
    yRatio = maxSize / h;
    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight])
      .div(255.0) // normalize
      .expandDims(0);
  });
    console.log(input);
    console.log(xRatio);
    console.log(yRatio);
    
    return [input, xRatio, yRatio];
}

const urlToTensor = async () => {
    // test on bus
    img = require('../assets/bus.jpg'); // replace with url
    imgAssetPath = Image.resolveAssetSource(img);
    const response = await fetch(imgAssetPath.uri, {}, { isBinary: true });
    const imgArrayBuffer = await response.arrayBuffer();
    const imgData = new Uint8Array(imgArrayBuffer);
    const imgTensor = decodeJpeg(imgData);

    return imgTensor;
}

export const detectNow = async (model, ctx) => {

    const numClass = labels.length;
    const img = await urlToTensor();
    console.log(img);
  
    const [modelHeight, modelWidth] = model.inputs[0].shape.slice(1,3);
    
    tf.engine().startScope();

    const [input, xRatio, yRatio] = await preprocess(img, modelWidth, modelHeight);

    const predictions = model.execute(input);
    
    const transRes = predictions.transpose([0, 2, 1]);

    const boxes = tf.tidy(() => {
      const w = transRes.slice([0, 0, 2], [-1, -1, 1]);
      const h = transRes.slice([0, 0, 3], [-1, -1, 1]);
      const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
      const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1

      return tf.concat([
        y1, x1, tf.add(y1, h), tf.add(x1, w)
      ], 2).squeeze();

    }) // get the boxes

    const [scores, classes] = tf.tidy(() => {
      // class scores
      const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0);
      return [rawScores.max(1), rawScores.argMax(1)];
    }) // get max scores and classes index

    const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes

    const boxes_data = boxes.gather(nms, 0).dataSync();
    const scores_data = scores.gather(nms, 0).dataSync();
    const classes_data = classes.gather(nms, 0).dataSync();

    console.log("Boxes--data");
    console.log(boxes_data);
    console.log("scores--data");
    console.log(scores_data);
    console.log("classes--data");
    console.log(classes_data);

    await renderBoxes(ctx, 0.1, boxes_data, scores_data, classes_data, [xRatio, yRatio]);

    tf.dispose([predictions, input, boxes, nms]); // clear memory

    tf.engine().endScope(); // end of scoping
}