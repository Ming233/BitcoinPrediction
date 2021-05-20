"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const usdSchema = new mongoose_1.Schema({
    date: {
        type: Date,
        required: false
    },
    value: {
        type: Number,
        required: false
    }
}, { timestamps: false });
exports.default = mongoose_1.model('Usd', usdSchema);
