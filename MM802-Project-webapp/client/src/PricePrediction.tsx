import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { Space, DatePicker, Divider, Checkbox } from 'antd';
import { RangeValue } from 'rc-picker/lib/interface';
import { getPricepredictions } from './api';
import moment from 'moment';
import './App.css';

interface IChartData {
  date: string;  
  price?: Number;
  lstm_predicted_price?: Number;
  bilstm_predicted_price?: Number;
  gru_predicted_price?: Number;
};

const { RangePicker } = DatePicker;

const PricePredictionPage: React.FC = () => {
  const [price, setPrice] = useState<boolean>(true);
  const [lstm_predicted_price, setLstm_predicted_price] = useState<boolean>(false);
  const [bilstm_predicted_price, setBilstm_predicted_price] = useState<boolean>(false);
  const [gru_predicted_price, setGru_predicted_price] = useState<boolean>(false);
  const [startDate, setStartDate] = useState<moment.Moment>(moment('2020-01-01'));
  const [endDate, setEndDate] = useState<moment.Moment>(moment('2020-12-31'));
  const [selectedData, setSelectedData] = useState<IChartData[]>([]);

  const loadData = useCallback(async () => {
    const prices = await getPricepredictions();
    const formattedUsds: IChartData[] = prices.map(u => { return { date: moment(u.date).format('YYYY-MM-DD'), price: u.price, lstm_predicted_price: u.lstm_predicted_price , bilstm_predicted_price: u.bilstm_predicted_price, gru_predicted_price: u.gru_predicted_price }; });
    let merged: IChartData[] = [];

    console.log(formattedUsds);
    for(let i = 0; i < formattedUsds.length; i++) {
      merged.push({
        ...formattedUsds[i], 
        ...(formattedUsds.find((itmInner) => itmInner.date === formattedUsds[i].date))}
      );
    }

    merged = merged.sort(sort);

    let dataBasedOnDate = merged.filter(d => moment(d.date).isSameOrAfter(startDate) && moment(d.date).isSameOrBefore(endDate));
    setSelectedData(dataBasedOnDate);
    
  }, [startDate, endDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="container-fluid">
      <h1>Featrue one - Price</h1>
      <div className="d-flex flex-row">
        <Space size={24}>
          <label className="mr-2">
            Select Date Range
          </label>
          <RangePicker
            showTime={false}
            format={'YYYY-MM-DD'}
            value={[startDate, endDate]}
            onChange={datesSelected}
          />
          <label className="mr-2">
            Select Data Type
          </label>
          <Checkbox onChange={onItemChecked} name="price" checked={price}>USD</Checkbox>
          <Checkbox onChange={onItemChecked} name="lstm_predicted_price" checked={lstm_predicted_price}>LSTM</Checkbox>
          <Checkbox onChange={onItemChecked} name="bilstm_predicted_price" checked={bilstm_predicted_price}>Bi-LSTM</Checkbox>
          <Checkbox onChange={onItemChecked} name="gru_predicted_price" checked={gru_predicted_price}>GRU</Checkbox>
        </Space>
      </div>
      <div className="d-inline-block">
        <LineChart
          width={1200}
          height={600}
          data={selectedData}
          className="height-fix"
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          {price && <Line type="monotone" dataKey="price" stroke="blue" />}
          {lstm_predicted_price && <Line type="monotone" dataKey="lstm_predicted_price" stroke="red" />}
          {bilstm_predicted_price && <Line type="monotone" dataKey="bilstm_predicted_price" stroke="green" />}
          {gru_predicted_price && <Line type="monotone" dataKey="gru_predicted_price" stroke="purple" />}
        </LineChart>
      </div>

      <Divider/>
    </div>
  );

  function onItemChecked(e: any) {
    if (e.target.name === 'price') {
      setPrice(!price);
    } else if (e.target.name === 'lstm_predicted_price') {
      setLstm_predicted_price(!lstm_predicted_price);
    } else if (e.target.name === 'bilstm_predicted_price') {
      setBilstm_predicted_price(!bilstm_predicted_price);
    } else if (e.target.name === 'gru_predicted_price') {
      setGru_predicted_price(!gru_predicted_price);
    } else {
      return;
    }
  }

  function datesSelected(dates: RangeValue<any>) {
    setStartDate(moment(dates![0]));
    setEndDate(moment(dates![1]));
	}

  function sort(a: IChartData, b: IChartData) {
    return moment(a.date).isAfter(moment(b.date)) ? 1 : -1;
  }

}

export default PricePredictionPage;
