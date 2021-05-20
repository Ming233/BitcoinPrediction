interface IHashRate {
    _id: string,
    date: string,
    value: Number
}

interface IUsd {
    _id: string,
    date: string,
    value: Number
}

interface IUsdprediction {
    _id: string,
    date: string,
    usd: Number,
    prediction: Number
}

interface ITweetprediction {
    _id: string,
    date: string,
    price: Number,
    lstm_predicted_price: Number,
    bilstm_predicted_price: Number,
    gru_predicted_price: Number
}

interface IPriceprediction {
    _id: string,
    date: string,
    price: Number,
    lstm_predicted_price: Number,
    bilstm_predicted_price: Number,
    gru_predicted_price: Number
}

interface IHrprediction {
    _id: string,
    date: string,
    price: Number,
    lstm_predicted_price: Number,
    bilstm_predicted_price: Number,
    gru_predicted_price: Number
}

interface IHrandpriceprediction {
    _id: string,
    date: string,
    price: Number,
    lstm_predicted_price: Number,
    bilstm_predicted_price: Number,
    gru_predicted_price: Number
}

interface IAllfeature {
    _id: string,
    date: string,
    price: Number,
    lstm_predicted_price: Number,
    bilstm_predicted_price: Number,
    gru_predicted_price: Number
}

interface IHashrateprediction {
    _id: string,
    date: string,
    hashrate: Number,
    prediction: Number
}

interface IKeyword {
    _id: string,
    word: String,
    count: Number
}

type ApiDataType = {
    message: string
    status: string
    hashRates: IHashRate[]
    usds: IUsd[]
    usdpredictions: IUsdprediction[]
    tweetpredictions: ITweetprediction[]
    pricepredictions: IPriceprediction[]
    hrpredictions: IHrprediction[]
    hrandpricepredictions: IHrandpriceprediction[]
    allfeatures: IAllfeature[]
    hashratepredictions: IHashrateprediction[]
    keywords: IKeyword[]
}
