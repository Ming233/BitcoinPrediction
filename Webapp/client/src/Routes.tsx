/**
 * Router, control the website direction
 * React file
 */
import * as React from "react";
import { Route, Switch } from "react-router-dom";
import MainPage from "./MainPage";
import HomePage from "./HomePage";
import usdandhr from "./USDandHR";
import usdprediction from "./PredictionUSD";
import tweetprediction from "./TweetPrediction";
import priceprediction from "./PricePrediction";
import hrprediction from "./HRPrediction";
import hrandpriceprediction from "./HRandpricePrediction";
import allfeatureprediction from "./AllFeaturePrediction";
import hashrateprediction from "./PredictionHashrate";
import hashratepredictiontwoline from "./PredictionHashrateTwoLines";
import textcloud from "./TextCloud";
import diagram from "./Diagram";
import paper from "./PaperPage";
import aboutproject from "./AboutProject";

class Routes extends React.Component<any, {}> {
  public render() {
    return (
      <React.Fragment>
        <Switch>
          <Route path="/main" component={MainPage} /> 
          <Route path="/home" component={HomePage} /> 
          <Route path="/usdandhr" component={usdandhr} /> 
          <Route path="/usdprediction" component={usdprediction} /> 
          <Route path="/tweetprediction" component={tweetprediction} /> 
          <Route path="/priceprediction" component={priceprediction} /> 
          <Route path="/HRPrediction" component={hrprediction} /> 
          <Route path="/hrandpriceprediction" component={hrandpriceprediction} /> 
          <Route path="/allfeatures" component={allfeatureprediction} /> 
          <Route path="/hashrateprediction" component={hashrateprediction} /> 
          <Route path="/hashratepredictiontwoline" component={hashratepredictiontwoline} /> 
          <Route path="/textcloud" component={textcloud} /> 
          <Route path="/diagram" component={diagram} /> 
          <Route path="/aboutproject" component={aboutproject} /> 
          <Route path="/paper" component={paper} /> 
          <Route path="*" component={MainPage} />
        </Switch>
      </React.Fragment>
    );
  }
}

export default Routes;