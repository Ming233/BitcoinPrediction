import { Router } from 'express'
import { getHashRates, getUsd, getUsdprediction, getHashrateprediction, getHrprediction, getPriceprediction, getTweetprediction, getKeyword, getHrandpriceprediction, getAllfeature } from '../controllers'
 
const router: Router = Router()

router.get('/api/hash-rates', getHashRates)

router.get('/api/usd', getUsd)

router.get('/api/usdprediction', getUsdprediction)

router.get('/api/hashrateprediction', getHashrateprediction)

router.get('/api/hrprediction', getHrprediction)

router.get('/api/priceprediction', getPriceprediction)

router.get('/api/tweetprediction', getTweetprediction)

router.get('/api/hrandpriceprediction', getHrandpriceprediction)

router.get('/api/allfeature', getAllfeature)

router.get('/api/keyword', getKeyword)

export default router
