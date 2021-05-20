"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const hrandpricepredictionSchema = new mongoose_1.Schema({
    date: {
        type: Date,
        required: false
    },
    price: {
        type: Number,
        required: false
    },
    lstm_predicted_price: {
        type: Number,
        required: false
    },
    bilstm_predicted_price: {
        type: Number,
        required: false
    },
    gru_predicted_price: {
        type: Number,
        required: false
    }
}, { timestamps: false });
exports.default = mongoose_1.model('Hrandpriceprediction', hrandpricepredictionSchema);
