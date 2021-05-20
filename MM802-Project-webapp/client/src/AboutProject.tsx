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
         The cryptocurrency market has developed rapidly in recent years and it has the trend of gradually replacing the traditional currency market. 
         Like stock price prediction, the traditional method uses past price to predict future price. Because the past price is independent of the future price, the result always unsatisfactory. 
         In this paper, we focus on using information related to the cryptocurrency market such as social media information to improve the price prediction model. 
         Our method is machine-learning-based and using high-frequency vocabulary gathered from social media, sentiment analysis of social media information, past price, past hash rate as prediction basis. 
         We use seven days as a time range to train our model and try to predict the eighth-day price increasing rate. We also tested our method on LSTM, bi-LSTM and GRU which are common stock price-prediction models. 
         Compare to the traditional stock price prediction, our cryptocurrency prediction method has better accuracy and timeliness.
        </Paragraph>
        <Title>Content</Title>
        <Divider/>
        <div className="mainpara">
          <Paragraph style={{marginRight: 250}}>
          With the price of cryptocurrencies (e.g. Bitcoin) skyrocketed, some machine learning and deep neural network architectures have been proposed for cryptocurrency analysis and prediction tasks. 
          However, many factors, like demand, policy, hash-rate, and more, cloud affect the price of cryptocurrencies. For many reasons, most cryptocurrency prediction algorithms are not accurate, leading to investors' loss. 
          The relationship between supply and demand generally determines the value of goods, and people are an essential factor in the relationship between supply and demand. 
          Our team decided to work on the natural language algorithm to analyze the relationship between bitcoin and language in this project. 
          Our team proposed three different advanced language algorithms LSTM, BI-LSTM, and GRU, to analyze the information from social media tools like Twitter and utilize the speech data to predict the price of Bitcoin. 
          All data is collected from the public community - Twitter. By process Twitter speech with these three modern algorithms, our team managed to forecast the price of cryptocurrencies. 
          In the end, a website will visualize those pre-processed data as graphical, user-friendly chat to the end-user. Our goal is to forecast the cryptocurrencies price trend for the following days through our pre-trained bitcoin price prediction model. 
          In addition, the internet user can visit our bitcoin prediction website to make up their decision.
          </Paragraph>
        </div>

        <Title>Tweet</Title>
        <Divider/>
        <div className="mainpara">
          <Title level={4}>Data Scraping</Title>
          <Paragraph>
            <li> <Link href="https://github.com/twintproject/twint" target="_Blank">Twint</Link>: A Python scraping library for collecting the Twitter users’ data.</li>
            <li> <Link href="https://www.nltk.org/" target="_Blank">NLTK</Link>: A language analysis library to convert tweets into formalized keywords.</li>
          </Paragraph>
          <Title level={4}>Sentiment Analysis</Title>
          <Paragraph>
            <li> <Link href="https://pypi.org/project/vaderSentiment/" target="_Blank">vaderSentiment</Link>: A language analysis library to analyze the sentiment intensity.</li>
          </Paragraph>
        </div>

        <Title>BTC Price Prediction</Title>
        <Divider/>
        <div className="mainpara">
          <Title level={4}>Data Normalization</Title>
          <Paragraph>
            <li> For some numerical variables, because their value ranges vary greatly, it is often necessary to organize their value ranges into a reasonable range.</li>
            <li> MinMaxScaler：<img src='minmax.png' /> </li> 
            <li>Min-Max only change the the value range. It does not change the distribution.</li>
            <img src='scaledPrice.jpg' />
          </Paragraph>
          <Title level={4}>Model</Title>
          <Paragraph>
            <li> Long short-term memory (<Link href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21" target="_Blank">LSTM</Link>)</li>
            <img src='lstm.png' width="400" height="400"/>

            <li> Gated Recurrent Units  (<Link href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21" target="_Blank">GRU</Link>)</li>
            <img src='gru.png' width="400" height="400"/>


            <li> Bidirectional Long short-term memory   (<Link href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21" target="_Blank">Bi-LSTM</Link>)</li>
            <img src='bilstm.png' width="600" height="400"/>

          </Paragraph>
        </div>
        
       
        <Title>NodeJS</Title>
        <Divider/>
        <div className="mainpara">
        <Title level={3}>Node</Title>
        <Paragraph>
          A JavaScript runtime built on Chrome's V8
          <li><Link href="https://nodejs.org/en/">Node official Website</Link></li>
        </Paragraph>
        </div>
        
        <Title>React App</Title>
        <Divider/>
        <div className="mainpara">
        <Title level={3}>React</Title>
        <Paragraph>
          React is a JavaScript library for building user interfaces.
          <li><Link href="https://reactjs.org/">React Document Website</Link></li>
          <li><Link href="https://github.com/facebook/react/">React Github</Link></li>
        </Paragraph>
        <Title level={3}>Ant Design</Title>
        <Paragraph>
          A design system for enterprise-level products. Create an efficient and enjoyable work experience.
          <li><Link href="https://ant.design/">Ant Design Website</Link></li>
          <li><Link href="https://github.com/ant-design/ant-design/">Ant Design Github</Link></li>
        </Paragraph>
        <Title level={3}>Recharts</Title>
        <Paragraph>
          Re-designed charting library built with React and D3.
          <li><Link href="https://recharts.org/en-US/">Rechart Website</Link></li>
          <li><Link href="https://github.com/recharts/recharts">Rechart Github</Link></li>
        </Paragraph>
        <Title level={3}>React-wordcloud</Title>
        <Paragraph>
          Simple React + D3 wordcloud component with powerful features. Uses the d3-cloud layout.
          <li><Link href="https://www.npmjs.com/package/react-wordcloud">react-wordcloud Website</Link></li>
          <li><Link href="https://github.com/chrisrzhou/react-wordcloud">React-wordcloud Github</Link></li>
        </Paragraph>
        <Title level={3}>React-Diagram</Title>
        <Paragraph>
          A flow & process orientated diagramming library inspired by Blender, Labview and Unreal engine.
          <li><Link href="https://github.com/projectstorm/react-diagrams">React-wordcloud Github</Link></li>
        </Paragraph>

        <Title level={5}>Other libraries</Title>
        <Paragraph>
          There are many libraries is used in this project.
          <li>Typescript</li>
          <li>Sass</li>
          <li>Moment</li>
        </Paragraph>

        </div>
        
        <Title>Data</Title>
        <Divider/>
        <div className="mainpara">
          <Title level={4}>Bit Coin Data</Title>
          <Paragraph>
            <li>Hash rate data from <Link href="https://www.quandl.com/data/BCHAIN-Blockchain" target="_Blank">quandl</Link> website</li>
          </Paragraph>
          <Title level={4}>Hash Rate</Title>
          <Paragraph>
            <li>USD data comes from  <Link href="https://www.quandl.com/data/BCHAIN-Blockchain" target="_Blank">quandl</Link> website.</li>          
          </Paragraph>
          <br/>
          <Paragraph>For more type of data, you can find it at <Link href=" ">University of Alberta</Link>.</Paragraph>
        </div>
        <Title>GitHub</Title>
        <Divider/>
        <div className="mainpara">
          <Title level={5}>React APP</Title>
          <Paragraph>
            <li>source code can be found at <Link href="https://github.com/Ming233/MM802Project">Git</Link></li>
          </Paragraph>
          <br/>
          <Title level={5}>Tweet</Title>
          <Paragraph>
            <li>source code can be found at <Link href="https://github.com/Ming233/MM802Project">Git</Link></li>
          </Paragraph>
          <br/>
          <Title level={5}>Bitcoin Prediction models application</Title>
          <Paragraph>
            <li>source code can be found at <Link href="https://github.com/Ming233/MM802Project">Git</Link></li>
          </Paragraph>
          <br/>
        </div>
        
      </Typography>
    </div>
  );
}


export default MainPage;