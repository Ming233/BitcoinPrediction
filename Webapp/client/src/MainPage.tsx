/**
 * HomePage Component 
 * shows the introduction and information about the website
 * Pure Html
 */
import React from "react";
import {Divider,Typography} from "antd";

const { Title, Paragraph, Link } = Typography;

const MainPage: React.FC = () => {
  return (
    <div className="center"> 
          
        <Title>Welcome to MM802 project</Title>
        <br/>
        <Typography>Social Media based Cryptocurrency Sentiment Analysis and Visualization</Typography>
        <br/>
        <br/>
        <img src='bitcoin.gif' width="200" height="200"  />
    </div>
  );
}


export default MainPage;