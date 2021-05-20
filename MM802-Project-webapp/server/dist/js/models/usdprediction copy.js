"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const usdpredictionSchema = new mongoose_1.Schema({
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
}, { timestamps: false });
exports.default = mongoose_1.model('Usdprediction', usdpredictionSchema);
