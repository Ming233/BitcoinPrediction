import React, { useState, useEffect, useCallback } from 'react';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/scale.css';
import './App.css';
import createEngine, {DefaultLinkModel, DefaultNodeModel, DiagramModel} from '@projectstorm/react-diagrams';
import {CanvasWidget} from '@projectstorm/react-canvas-core';

  // create an instance of the engine with all the defaults
  const engine = createEngine();

  const mangoDB = new DefaultNodeModel({
      name: 'MongoDB',
      color: 'rgb(123, 0, 255)',
  });
  mangoDB.setPosition(600, 50);
  let mangoDBport = mangoDB.addOutPort('Database');

  const Twint = new DefaultNodeModel({
      name: 'Twint',
      color: 'rgb(111, 255, 0)',
  });
  Twint.setPosition(1000, 100);
  let Twintport = Twint.addOutPort('Scraping');

  const twitter = new DefaultNodeModel({
      name: 'Twitter',
      color: 'rgb(0, 192, 255)',
  });
  twitter.setPosition(1050, 450);
  let twitterport = twitter.addOutPort('Tweet');

  const lstm = new DefaultNodeModel({
      name: 'LSTM',
      color: 'rgb(245, 227, 171)',
  });
  lstm.setPosition(400, 300);
  let lstmport = lstm.addOutPort('Data analyze');

  const bilist = new DefaultNodeModel({
      name: 'GRU',
      color: 'rgb(171, 230, 245)',
  });
  bilist.setPosition(800, 300);
  let bilistport = bilist.addOutPort('Data analyze');

  const bilstm = new DefaultNodeModel({
      name: 'Bi-LSTM',
      color: 'rgb(153, 240, 172)',
  });
  bilstm.setPosition(600, 500);
  let bilstmport = bilstm.addOutPort('Data analyze');

  const NodeJS = new DefaultNodeModel({
      name: 'NodeJS',
      color: 'rgb(175,154,245)',
  });
  NodeJS.setPosition(400, 150);
  let NodeJSport = NodeJS.addOutPort('REST API');

  const react = new DefaultNodeModel({
      name: 'React',
      color: 'rgb(224, 245, 154)',
  });
  react.setPosition(100, 100);
  let reactport = react.addOutPort('data visualization');

  const enduser = new DefaultNodeModel({
      name: 'User',
      color: 'rgb(245, 154, 195)',
  });
  enduser.setPosition(100, 350);
  let enduserport = enduser.addOutPort('Visit Website');

  const link1 = Twintport.link<DefaultLinkModel>(mangoDBport);
  link1.addLabel('Input data');

  const link2 = twitterport.link<DefaultLinkModel>(Twintport);
  link2.addLabel('Collect Tweet');

  const link3a = mangoDBport.link<DefaultLinkModel>(bilstmport);
  link3a.addLabel('Data Process');

  const link3b = mangoDBport.link<DefaultLinkModel>(lstmport);
  link3b.addLabel('Data Process');

  const link3c = mangoDBport.link<DefaultLinkModel>(bilistport);
  link3c.addLabel('Data Process');

  const link4a = mangoDBport.link<DefaultLinkModel>(NodeJSport);
  link4a.addLabel('Request Data')

  
  const link4b = NodeJSport.link<DefaultLinkModel>(reactport);
  link4b.addLabel('Convert JSON')

  const link5 = reactport.link<DefaultLinkModel>(enduserport);
  link5.addLabel('Generate Graph')
  
  const model = new DiagramModel();
  model.addAll(mangoDB, twitter, Twint, lstm, bilist, bilstm, react, NodeJS, enduser);
  model.addAll(link1,link2, link3a, link3b, link3c, link4a, link4b, link5);
  engine.setModel(model);

const Diagram: React.FC = () => {
  useEffect(() => {
    engine.setModel(model)
  }, [])

  useEffect(() => {
  return () => {
      engine.setModel(new DiagramModel())
    }
  }, [])

  return (
    <div className="container-fluid">
      <h1> Diagram </h1>
      <CanvasWidget className="reactdiagram" engine={engine} />
    </div>
  );
}

export default Diagram;