import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Image } from 'react-native';

import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

export default function App() {

  const [isTfReady, setIsTfReady] = useState(false);
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
      
      const dummyInput = tf.ones(model.inputs[0].shape);
      const warmup = model.execute(dummyInput);
      setIsTfReady(true);

      tf.dispose([warmup, dummyInput]); // cleanup memory

    } catch (err) {
      console.log(err);
    }
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
});
