import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { Space, DatePicker, Divider, Checkbox } from 'antd';
import { RangeValue } from 'rc-picker/lib/interface';
import { getHashratepredictions } from './api';
import moment from 'moment';
import './App.css';

interface IChartData {
  date: string;
  hashrate?: Number;
  usd?: Number;
  hashrateprediction?: Number;
  usdprediction?: Number;
};

const { RangePicker } = DatePicker;

const BitCoinHRPage: React.FC = () => {
  const [hashrate, setHashrate] = useState<boolean>(true);
  const [hashrateprediction, setHashrateprediction] = useState<boolean>(false);
  const [startDate, setStartDate] = useState<moment.Moment>(moment('2020-01-01'));
  const [endDate, setEndDate] = useState<moment.Moment>(moment('2021-01-15'));
  const [selectedData, setSelectedData] = useState<IChartData[]>([]);

  const loadData = useCallback(async () => {
    const hashrates = await getHashratepredictions();
    const formattedHashrates: IChartData[] = hashrates.map(hr => { return { date: moment(hr.date).format('YYYY-MM-DD'), hashrate: hr.hashrate, hashrateprediction: hr.prediction }; });
    let merged: IChartData[] = [];

    for(let i = 0; i < formattedHashrates.length; i++) {
      merged.push({
        ...formattedHashrates[i], 
        ...(formattedHashrates.find((itmInner) => itmInner.date === formattedHashrates[i].date))}
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
      <h1>Bitcoin Hash Rate Trend</h1>
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
          <Checkbox onChange={onItemChecked} name="hashrate" checked={hashrate}>Hash Rate</Checkbox>
          <Checkbox onChange={onItemChecked} name="hashrateprediction" checked={hashrateprediction}>USD Prediction</Checkbox>
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
          {hashrate && <Line type="monotone" dataKey="hashrate" stroke="green" />}
          {hashrateprediction && <Line type="monotone" dataKey="hashrateprediction" stroke="black" />}
        </LineChart>
      </div>
      {/* <Space size={24}>
        <div>Keyword count: TODO</div>
        <div>TODO B</div>
        <div>TODO C</div>
      </Space> */}
      <Divider/>
    </div>
  );

  function onItemChecked(e: any) {
    if (e.target.name === 'hashrate') {
      setHashrate(!hashrate);
    } else if (e.target.name === 'hashrateprediction') {
      setHashrateprediction(!hashrateprediction);
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

export default BitCoinHRPage;
