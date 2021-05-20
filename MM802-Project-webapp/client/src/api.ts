import axios, { AxiosResponse } from "axios";

const apiUrl: string = "http://localhost:4000/api";

export const getHashRates = async (): Promise<IHashRate[]> => {
  try {
    const hashRatesResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/hash-rates");
    return hashRatesResponse.data.hashRates;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getUsds = async (): Promise<IUsd[]> => {
  try {
    const usdsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/usd");
    return usdsResponse.data.usds;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getUsdpredictions = async (): Promise<IUsdprediction[]> => {
  try {
    const usdpredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/usdprediction");
    return usdpredictionsResponse.data.usdpredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getTweetpredictions = async (): Promise<ITweetprediction[]> => {
  try {
    const tweetpredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/tweetprediction");
    return tweetpredictionsResponse.data.tweetpredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getPricepredictions = async (): Promise<IPriceprediction[]> => {
  try {
    const pricepredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/priceprediction");
    return pricepredictionsResponse.data.pricepredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getHrpredictions = async (): Promise<IHrprediction[]> => {
  try {
    const hrpredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/hrprediction");
    return hrpredictionsResponse.data.hrpredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getHrandpricepredictions = async (): Promise<IHrandpriceprediction[]> => {
  try {
    const hrandpricepredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/hrandpriceprediction");
    return hrandpricepredictionsResponse.data.hrandpricepredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getAllfeatures = async (): Promise<IAllfeature[]> => {
  try {
    const allfeaturesResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/allfeature");
    return allfeaturesResponse.data.allfeatures;
  } catch (err) {
    console.log(err);
    throw err;
  }
}

export const getHashratepredictions = async (): Promise<IHashrateprediction[]> => {
  try {
    const hashratepredictionsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/hashrateprediction");
    return hashratepredictionsResponse.data.hashratepredictions;
  } catch (err) {
    console.log(err);
    throw err;
  }
}


export const getKeywords = async (): Promise<IKeyword[]> => {
  try {
    const keywordsResponse: AxiosResponse<ApiDataType> = await axios.get(apiUrl + "/keyword");
    return keywordsResponse.data.keywords;
  } catch (err) {
    console.log(err);
    throw err;
  }
}