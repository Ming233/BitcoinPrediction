import { IUsdprediction } from '../types/types';
import { model, Schema } from 'mongoose'

const usdpredictionSchema: Schema = new Schema({
    date: {
        type: Date,
        required: false
    },

    usd: {
        type: Number,
        required: false
    },

    prediction: {
        type: Number,
        required: false
    }

}, { timestamps: false })


export default model<IUsdprediction>('Usdprediction', usdpredictionSchema)