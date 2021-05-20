import { Document } from 'mongoose'

export interface IHashRate extends Document {
    date: Date
    value: Number
}

export interface IUsd extends Document {
    date: Date
    value: Number
}

export interface IUsdprediction extends Document {
    date: Date
    usd: Number
    prediction: Number
}

export interface IHashrateprediction extends Document {
    date: Date
    hashrate: Number
    prediction: Number
}

export interface IHrprediction extends Document {
    date: Date
    price: Number
    lstm_predicted_price: Number
    bilstm_predicted_price: Number
    gru_predicted_price: Number
}

export interface IPriceprediction extends Document {
    date: Date
    price: Number
    lstm_predicted_price: Number
    bilstm_predicted_price: Number
    gru_predicted_price: Number
}

export interface ITweetprediction extends Document {
    date: Date
    price: Number
    lstm_predicted_price: Number
    bilstm_predicted_price: Number
    gru_predicted_price: Number
}


export interface IHrandpriceprediction extends Document {
    date: Date
    price: Number
    lstm_predicted_price: Number
    bilstm_predicted_price: Number
    gru_predicted_price: Number
}


export interface IAllfeature extends Document {
    date: Date
    price: Number
    lstm_predicted_price: Number
    bilstm_predicted_price: Number
    gru_predicted_price: Number
}

export interface IKeyword extends Document {
    word: String
    count: Number
}