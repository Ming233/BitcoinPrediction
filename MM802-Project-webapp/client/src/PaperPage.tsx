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
    <div>      
      <Typography style={{marginLeft: 15}}>
        <Title>Introduction</Title>
        <Divider/>
        <Paragraph className="mainpara" style={{marginRight: 250}}>
          Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. Some text go here. 
        </Paragraph>
        <Title>Content</Title>
        <Divider/>
        <div className="mainpara">
          <Paragraph style={{marginRight: 250}}>
          Paper go here.
           <Link href=" "> document</Link>.
          </Paragraph>
        </div>
        
        <Title>Data</Title>
        <Divider/>
        <div className="mainpara">
          <Title level={4}>Bit Coin Data</Title>
          <Paragraph>
            <li>hash rate <Link href=" ">data</Link> from xxx website</li>
          </Paragraph>
          <Title level={4}>Hash Rate</Title>
          <Paragraph>
            <li>USD <Link href=" ">data</Link> from xxx website.</li>          
          </Paragraph>
          <Title level={4}>More data</Title>
          <Paragraph>
            <li>The analyzed data <Link href=" /">Data Source</Link> from xxx paper.</li>          
          </Paragraph>
          <br/>
          <Paragraph>For more type of data, you can find it at <Link href=" ">University of Alberta</Link>.</Paragraph>
        </div>
        <Title>Packages</Title>
        <Divider/>
        <div className="mainpara">
        <Title level={3}>React</Title>
        <Paragraph>
          <li>React is a JavaScript library for building user interfaces.</li>
          <li><Link href="https://reactjs.org/">React Document Website</Link></li>
          <li><Link href="https://github.com/facebook/react/">React Github</Link></li>
        </Paragraph>
        <Title level={3}>Ant Design</Title>
        <Paragraph>
          <li>A design system for enterprise-level products. Create an efficient and enjoyable work experience.</li>
          <li><Link href="https://ant.design/">Ant Design Website</Link></li>
          <li><Link href="https://github.com/ant-design/ant-design/">Ant Design Github</Link></li>
        </Paragraph>
        <Title level={3}>Recharts</Title>
        <Paragraph>
          <li>Re-designed charting library built with React and D3.</li>
          <li><Link href="https://recharts.org/en-US/">Rechart Website</Link></li>
          <li><Link href="https://github.com/recharts/recharts">Rechart Github</Link></li>
        </Paragraph>
        </div>
      </Typography>
    </div>
  );
}


export default MainPage;