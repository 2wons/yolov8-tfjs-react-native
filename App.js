
import { StyleSheet, Text, View } from 'react-native';

import React, { useEffect, useState, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

import { TouchableOpacity } from 'react-native';

import { GLView } from "expo-gl";
import Expo2DContext from "expo-2d-context";
import { detectNow } from './utils/detect';


export default function App() {

  const ctxRef = useRef(null);

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

  const predictnow = async () => {
    await detectNow(model, ctxRef.current);
  }

  const handleSetup = useCallback((gl) => {
    const ctx = new Expo2DContext(gl);
    ctxRef.current = ctx;
    ctx.fillStyle = 'grey';
    ctx.fillRect(100,100,100,100);
    ctx.flush();
  }, [])

  useEffect(() => {
    load();//
    if (ctxRef.current) {
      ctxRef.current.clearRect(0,0,640,640);
      ctxRef.current.fillStyle = "grey";
      ctxRef.current.fillRect(100,150,50,50);
      ctxRef.current.fillRect(100,150,50,50);
      ctxRef.current.flush();
      console.log('thre should be drawings');
    }
  }, []);

  return (
    <View style={styles.container}>
      <Text>Open up App.js to start working on your app!</Text>
      {isTfReady ? (
        <Text>Tf is ready!</Text>
      ) : null }
     <TouchableOpacity onPress={predictnow}>
          <View style={styles.button}>
            <Text style={styles.buttonText}>Predict</Text>
          </View>
      </TouchableOpacity>
      <View style={styles.bboxView}>
          <GLView
            onContextCreate={handleSetup}
          />
      </View>
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
  bboxView: {
    width: 700,
    height: 700,
    justifyContent: 'center',
    alignContent: 'center'
    //backgroundColor: 'grey'
  },
  canvas: {
    position: 'absolute',
    width: 640,
    height: 640,
    zIndex: 99,
  }
});
