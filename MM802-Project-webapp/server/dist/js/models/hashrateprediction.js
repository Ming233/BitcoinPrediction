"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const hashratepredictionSchema = new mongoose_1.Schema({
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
}, { timestamps: false });
exports.default = mongoose_1.model('Hashrateprediction', hashratepredictionSchema);
