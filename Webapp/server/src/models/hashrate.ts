import { IHashRate } from './../types/types';
import { model, Schema } from 'mongoose'

const hashRateSchema: Schema = new Schema({
    date: {
        type: Date,
        required: false
    },

    value: {
        type: Number,
        required: false
    }

}, { timestamps: false })


export default model<IHashRate>('Hashrate', hashRateSchema)