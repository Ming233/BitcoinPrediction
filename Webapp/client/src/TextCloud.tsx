import React, { useState, useEffect, useCallback } from 'react';
import ReactWordcloud from 'react-wordcloud';
import 'tippy.js/dist/tippy.css';
import 'tippy.js/animations/scale.css';
import { getKeywords } from './api';
import moment from 'moment';
import './App.css';
import {Word, MinMaxPair} from 'react-wordcloud/types';
import { Resizable } from "re-resizable";

interface IChartData {
  text: String;
  value: Number;
};

const Wordcloud: React.FC = () => {
  const [selectedData, setSelectedData] = useState<Word[]>([]);

  const loadData = useCallback(async () => {
    const keywords = await getKeywords();
    const formattedKeywords: Word[] = keywords.map(kw => { return { text: kw.word.toString(), value: +kw.count }; });

    setSelectedData(formattedKeywords);
    
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const size:MinMaxPair = [1200, 600];
  const fontSizes:MinMaxPair = [20, 88];
  const options = {
    fontSizes
  };

  const resizeStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  border: "solid 1px #ddd",
  background: "#f0f0f0"
  };


  return (
    <div className="container-fluid">
      <h1>Most Popular Texts</h1>
      {/* @ts-ignore */}
      <Resizable defaultSize={{width: 600, height: 300}} style={resizeStyle} >
        <div style={{ width: "100%", height: "100%" }}>
        <ReactWordcloud words={selectedData} options={options} />
      </div>
      </Resizable>
    </div>
  );




}


export default Wordcloud;