"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const keywordSchema = new mongoose_1.Schema({
    word: {
        type: String,
        required: false
    },
    count: {
        type: Number,
        required: false
    }
}, { timestamps: false });
exports.default = mongoose_1.model('Keyword', keywordSchema);
