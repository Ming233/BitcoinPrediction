"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getKeyword = exports.getAllfeature = exports.getHrandpriceprediction = exports.getTweetprediction = exports.getPriceprediction = exports.getHrprediction = exports.getHashrateprediction = exports.getUsdprediction = exports.getUsd = exports.getHashRates = void 0;
const hashrate_1 = __importDefault(require("../models/hashrate"));
const usd_1 = __importDefault(require("../models/usd"));
const usdprediction_1 = __importDefault(require("../models/usdprediction"));
const hashrateprediction_1 = __importDefault(require("../models/hashrateprediction"));
const hrprediction_1 = __importDefault(require("../models/hrprediction"));
const priceprediction_1 = __importDefault(require("../models/priceprediction"));
const tweetprediction_1 = __importDefault(require("../models/tweetprediction"));
const hrandpriceprediction_1 = __importDefault(require("../models/hrandpriceprediction"));
const allfeature_1 = __importDefault(require("../models/allfeature"));
const keyword_1 = __importDefault(require("../models/keyword"));
const getHashRates = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const hashRates = yield hashrate_1.default.find();
        res.status(200).json({ hashRates });
    }
    catch (error) {
        throw error;
    }
});
exports.getHashRates = getHashRates;
const getUsd = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const usds = yield usd_1.default.find();
        res.status(200).json({ usds });
    }
    catch (error) {
        throw error;
    }
});
exports.getUsd = getUsd;
const getUsdprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const usdpredictions = yield usdprediction_1.default.find();
        res.status(200).json({ usdpredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getUsdprediction = getUsdprediction;
const getHashrateprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const hashratepredictions = yield hashrateprediction_1.default.find();
        res.status(200).json({ hashratepredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getHashrateprediction = getHashrateprediction;
const getHrprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const hrpredictions = yield hrprediction_1.default.find();
        res.status(200).json({ hrpredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getHrprediction = getHrprediction;
const getPriceprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const pricepredictions = yield priceprediction_1.default.find();
        res.status(200).json({ pricepredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getPriceprediction = getPriceprediction;
const getTweetprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const tweetpredictions = yield tweetprediction_1.default.find();
        res.status(200).json({ tweetpredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getTweetprediction = getTweetprediction;
const getHrandpriceprediction = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const hrandpricepredictions = yield hrandpriceprediction_1.default.find();
        res.status(200).json({ hrandpricepredictions });
    }
    catch (error) {
        throw error;
    }
});
exports.getHrandpriceprediction = getHrandpriceprediction;
const getAllfeature = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const allfeatures = yield allfeature_1.default.find();
        res.status(200).json({ allfeatures });
    }
    catch (error) {
        throw error;
    }
});
exports.getAllfeature = getAllfeature;
const getKeyword = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const keywords = yield keyword_1.default.find();
        res.status(200).json({ keywords });
    }
    catch (error) {
        throw error;
    }
});
exports.getKeyword = getKeyword;
