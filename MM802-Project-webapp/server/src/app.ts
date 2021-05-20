import express, { Express } from 'express'
import mongoose, { isValidObjectId } from 'mongoose'
import cors from 'cors'
import apiRoutes from './routes'

const app: Express = express()

const PORT: string | number = process.env.PORT || 4000

app.use(express.urlencoded({ extended: true }))
app.use(express.json())
app.use(cors())
app.use(apiRoutes)

const uri: string = `mongodb+srv://${process.env.MONGO_USER}:${process.env.MONGO_PASSWORD}@cluster0.plf4b.mongodb.net/${process.env.MONGO_DB}?retryWrites=true&w=majority`
const options = { useNewUrlParser: true, useUnifiedTopology: true }
mongoose.set('useFindAndModify', false)

mongoose
    .connect(uri, options)
    .then(() =>
        app.listen(PORT,
            async () => {
                console.log(`Server running on http://localhost:${PORT}`)
            }
        )
    )
    .catch((error) => {
        throw error
    })
