import './App.css';
import Routes from './Routes';
import { BrowserRouter as Router, Route, Link } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import React, { useState } from "react";
import { HomeOutlined, DollarOutlined, CalculatorOutlined, LineChartOutlined, CommentOutlined, AlignCenterOutlined, ReadOutlined, AliwangwangOutlined, TeamOutlined, BoldOutlined, SettingOutlined  } from '@ant-design/icons';

interface IProps {
  // history?: any;
}

const App:React.FC<IProps>=(props)=> {
  const [current, setCurrent] = useState<string>('mail');

  const { Header, Content, Footer, Sider } = Layout;
  const { SubMenu } = Menu;

  return (
    <Router>
      <div>
        {/* @ts-ignore */}
        <Menu onClick={(e)=>{setCurrent(e.key); console.log(e.key)}} selectedKeys={[current]} mode="horizontal">
        <img src="UofAlogo.jpg"  width="180" height="40"/>
          <Menu.Item key="mail" icon={<HomeOutlined />}>
            <Link to="/main">HOME</Link>
          </Menu.Item>
          <Menu.Item key="usdprediction" icon={<DollarOutlined />}>
            <Link to="/usdprediction">USD prediction</Link>
          </Menu.Item>
          <SubMenu key="SubMenu" icon={<BoldOutlined />} title="Four Features">
              <Menu.ItemGroup title="Feature One">
                <Menu.Item key="setting:1">
                  <Link to="/priceprediction">Price</Link></Menu.Item>
              </Menu.ItemGroup>
              <Menu.ItemGroup title="Feature Two">
                <Menu.Item key="setting:2">
                  <Link to="/hrprediction">Hashrate</Link></Menu.Item>
              </Menu.ItemGroup>
              <Menu.ItemGroup title="Feature Three">
                <Menu.Item key="setting:3">
                  <Link to="/HRandpricePrediction">Price and Hashrate</Link>
                </Menu.Item>
              </Menu.ItemGroup>
              <Menu.ItemGroup title="Feature Four">
                <Menu.Item key="setting:4">
                  <Link to="/allfeatures">Price, HR and Tweet</Link></Menu.Item>
              </Menu.ItemGroup>
          </SubMenu>
          <Menu.Item key="hashrateprediction" icon={<CalculatorOutlined />}>
            <Link to="/hashrateprediction">Hashrate prediction</Link>
          </Menu.Item>
          <Menu.Item key="twoline" icon={<LineChartOutlined />}>
            <Link to="/hashratepredictiontwoline">Two Line</Link>
          </Menu.Item>
          <Menu.Item key="textcloud" icon={<CommentOutlined />}>
            <Link to="/textcloud">Text Cloud</Link>
          </Menu.Item>
          <Menu.Item key="diagram" icon={<AlignCenterOutlined />}>
            <Link to="/diagram">Diagram</Link>
          </Menu.Item>
          <Menu.Item key="aboutproject" icon={<TeamOutlined />}>
            <Link to="/aboutproject">About this Project</Link>
          </Menu.Item>
          <Menu.Item key="alipay4" icon={<ReadOutlined />}>
            <a href="https://drive.google.com/file/d/1aDE3TV34l9V68ZczVi0DKV1g6FI6r4ff/view?usp=sharing" target="_blank" rel="noopener noreferrer">Paper</a>
          </Menu.Item>
          <Menu.Item key="alipay5" icon={<AliwangwangOutlined />}>
            <a href="https://drive.google.com/file/d/1LXQjXpKedaSHQ3ukGYEa72v95_DvCG7U/view?usp=sharing" target="_blank" rel="noopener noreferrer">Presentation</a>
          </Menu.Item>
        </Menu>
        <Route component={Routes}/>
        
        <Footer style={{ textAlign: 'right' }}>University of Alberta MRC ©2021 <br/>Created by Ningbo Zhu, Fei Yang, and Mingzhi Zhu.</Footer>
      </div>
    </Router>
  );
}

export default App;