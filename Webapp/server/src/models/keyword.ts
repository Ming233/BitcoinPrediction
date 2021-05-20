import { IKeyword } from '../types/types';
import { model, Schema } from 'mongoose'

const keywordSchema: Schema = new Schema({
    word: {
        type: String,
        required: false
    },

    count: {
        type: Number,
        required: false
    }

}, { timestamps: false })


export default model<IKeyword>('Keyword', keywordSchema)