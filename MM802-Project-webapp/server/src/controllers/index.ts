import { Response, Request } from 'express'
import { IHashRate, IUsd, IUsdprediction, IHashrateprediction, IHrprediction, IPriceprediction, ITweetprediction, IHrandpriceprediction, IKeyword } from './../types/types'
import HashRate from '../models/hashrate'
import Usd from '../models/usd'
import Usdprediction from '../models/usdprediction'
import Hashrateprediction from '../models/hashrateprediction'
import Hrprediction from '../models/hrprediction'
import Priceprediction from '../models/priceprediction'
import Tweetprediction from '../models/tweetprediction'
import Hrandpriceprediction from '../models/hrandpriceprediction'
import Allfeature from '../models/allfeature'
import Keyword from '../models/keyword'

const getHashRates = async (req: Request, res: Response): Promise<void> => {
    try {
        const hashRates: IHashRate[] = await HashRate.find()
        res.status(200).json({ hashRates })
    } catch (error) {
        throw error
    }
}

const getUsd = async (req: Request, res: Response): Promise<void> => {
    try {
        const usds: IUsd[] = await Usd.find()
        res.status(200).json({ usds })
    } catch (error) {
        throw error
    }
}

const getUsdprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const usdpredictions: IUsdprediction[] = await Usdprediction.find()
        res.status(200).json({ usdpredictions })
    } catch (error) {
        throw error
    }
}

const getHashrateprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const hashratepredictions: IHashrateprediction[] = await Hashrateprediction.find()
        res.status(200).json({ hashratepredictions })
    } catch (error) {
        throw error
    }
}

const getHrprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const hrpredictions: IHrprediction[] = await Hrprediction.find()
        res.status(200).json({ hrpredictions })
    } catch (error) {
        throw error
    }
}

const getPriceprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const pricepredictions: IPriceprediction[] = await Priceprediction.find()
        res.status(200).json({ pricepredictions })
    } catch (error) {
        throw error
    }
}

const getTweetprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const tweetpredictions: ITweetprediction[] = await Tweetprediction.find()
        res.status(200).json({ tweetpredictions })
    } catch (error) {
        throw error
    }
}

const getHrandpriceprediction = async (req: Request, res: Response): Promise<void> => {
    try {
        const hrandpricepredictions: IHrandpriceprediction[] = await Hrandpriceprediction.find()
        res.status(200).json({ hrandpricepredictions })
    } catch (error) {
        throw error
    }
}

const getAllfeature = async (req: Request, res: Response): Promise<void> => {
    try {
        const allfeatures: ITweetprediction[] = await Allfeature.find()
        res.status(200).json({ allfeatures })
    } catch (error) {
        throw error
    }
}

const getKeyword = async (req: Request, res: Response): Promise<void> => {
    try {
        const keywords: IKeyword[] = await Keyword.find()
        res.status(200).json({ keywords })
    } catch (error) {
        throw error
    }
}


export { getHashRates, getUsd, getUsdprediction, getHashrateprediction, getHrprediction, getPriceprediction, getTweetprediction, getHrandpriceprediction, getAllfeature, getKeyword }
