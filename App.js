import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Image } from 'react-native';

import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system'
import { TouchableOpacity } from 'react-native';
import labels from "./labels.json"

function preprocess(img, modelWidth, modelHeight) {
  let widthRatio, heightRatio;

  const input = tf.tidy(() => {
    
    const [h, w] = img.shape.slice(0, 2);
    const maxSize = Math.max(w, h);
    const imgPadded = img.pad([
      [0, maxSize - h],
      [0, maxSize - w],
      [0, 0]
    ]);
    widthRatio = maxSize / w;
    heightRatio = maxSize / h;
    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight])
      .div(255.0)
      .expandDims(0);
  });
  return [input, widthRatio, heightRatio];
}

async function detect(source, model){
  const [modelWidth, modelHeight] = model.inputShape.slice(1,3);
  tf.engine().startScope();

  const [input, widthRatio, heightRatio] = preprocess(source, modelWidth, modelHeight);
  const predictions = model.network.execute(input);

}


async function imgToTensor(uri) {
  const img64 = await FileSystem.readAsStringAsync(uri, {
    encoding:FileSystem.EncodingType.Base64
  });
  
  const imgBuffer = tf.util.encodeString(img64, 'base64').buffer
  const raw = new Uint8Array(imgBuffer);
  let imgTensor = decodeJpeg(raw);

  return imgTensor;
}

async function urlToTensor(url) {
  // test on bus
  img = require('./assets/bus.jpg'); // replace with url
  imgAssetPath = Image.resolveAssetSource(img);
  const response = await fetch(imgAssetPath.uri, {}, { isBinary: true });
  const imgArrayBuffer = await response.arrayBuffer();
  const imgData = new Uint8Array(imgArrayBuffer);
  const imgTensor = decodeJpeg(imgData);

  return imgTensor;
}

export default function App() {

  const [isTfReady, setIsTfReady] = useState(false);
  let [model, setModel] = useState(null);
  modelJSON = require('./assets/yolov8n_web_model/model.json');
  modelWeights = [
    require('./assets/yolov8n_web_model/group1-shard1of4.bin'),
    require('./assets/yolov8n_web_model/group1-shard2of4.bin'),
    require('./assets/yolov8n_web_model/group1-shard3of4.bin'),
    require('./assets/yolov8n_web_model/group1-shard4of4.bin'),
  ];

  const modelURI = bundleResourceIO(modelJSON, modelWeights);

  const load = async () => {
    try {
      await tf.ready(); 
      model = await tf.loadGraphModel(modelURI)
      
      // warmup the model
      const dummyInput = tf.ones(model.inputs[0].shape);
      const warmup = model.execute(dummyInput);
      setIsTfReady(true);
      setModel(model);

      tf.dispose([warmup, dummyInput]); // cleanup memory

    } catch (err) {
      console.log(err);
    }
  }

  const testPredict = async () => {
    //console.log(model);
    const numClass = labels.length;
    const img = await urlToTensor();
    console.log(model.inputs[0].shape)
  
    const [modelHeight, modelWidth] = model.inputs[0].shape.slice(1,3);
    

    tf.engine().startScope();
    const [input, widthRatio, heightRatio] = preprocess(img, modelWidth, modelHeight);
    const predictions = model.execute(input);
    console.log(input);
    
    const transRes = predictions.transpose([0, 2, 1]);
    console.log(transRes);

    const boxes = tf.tidy(() => {
      const w = transRes.slice([0, 0, 2], [-1, -1, 1]);
      const h = transRes.slice([0, 0, 3], [-1, -1, 1]);
      const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
      const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1

      return tf.concat([
        y1, x1, tf.add(y1, h), tf.add(x1, w)
      ], 2).squeeze();

    }) // get the boxes
    console.log("---------Logging boxes-------");
    console.log(boxes);

    const [scores, classes] = tf.tidy(() => {
      // class scores
      const rawScores = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze(0);
      return [rawScores.max(1), rawScores.argMax(1)];
    }) // get max scores and classes index

    console.log("NMS Suppression")
    const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2); // NMS to filter boxes
    console.log(nms);

    const boxes_data = boxes.gather(nms, 0).dataSync();
    const scores_data = scores.gather(nms, 0).dataSync();
    const classes_data = classes.gather(nms, 0).dataSync();

    console.log("Boxes--data");
    console.log(boxes_data);
    console.log("scores--data");
    console.log(scores_data);
    console.log("classes--data");
    console.log(classes_data);

    tf.dispose([predictions, input, boxes, nms]); // clear memory

    tf.engine().endScope(); // end of scoping
    
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <View style={styles.container}>
      <Text>Open up App.js to start working on your app!</Text>
      {isTfReady ? (
        <Text>Tf is ready!</Text>
      ) : null }
     <TouchableOpacity onPress={testPredict}>
          <View style={styles.button}>
            <Text style={styles.buttonText}>Predict</Text>
          </View>
      </TouchableOpacity>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    marginBottom: 30,
    width: 260,
    alignItems: 'center',
    backgroundColor: '#2196F3',
  },
  buttonText: {
    textAlign: 'center',
    padding: 20,
    color: 'white',
  },
});
