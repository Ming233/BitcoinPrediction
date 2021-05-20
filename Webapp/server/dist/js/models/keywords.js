"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mongoose_1 = require("mongoose");
const keywordSchema = new mongoose_1.Schema({
    Date: {
        type: String
    },
    Count: {
        type: String
    },
    Keyword: {
        type: Boolean
    }
});
exports.default = mongoose_1.model('Keywords', keywordSchema);
