import { IHashrateprediction } from '../types/types';
import { model, Schema } from 'mongoose'

const hashratepredictionSchema: Schema = new Schema({
    date: {
        type: Date,
        required: false
    },

    hashrate: {
        type: Number,
        required: false
    },

    prediction: {
        type: Number,
        required: false
    }

}, { timestamps: false })


export default model<IHashrateprediction>('Hashrateprediction', hashratepredictionSchema)