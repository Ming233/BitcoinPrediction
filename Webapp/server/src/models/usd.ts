import { IUsd } from './../types/types';
import { model, Schema } from 'mongoose'

const usdSchema: Schema = new Schema({
    date: {
        type: Date,
        required: false
    },

    value: {
        type: Number,
        required: false
    }

}, { timestamps: false })


export default model<IUsd>('Usd', usdSchema)