import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { Space, DatePicker, Divider, Checkbox } from 'antd';
import { RangeValue } from 'rc-picker/lib/interface';
import { getHashRates, getUsds } from './api';
import moment from 'moment';
import './App.css';

interface IChartData {
  date: string;
  hashrate?: Number;
  usd?: Number;
};

const { RangePicker } = DatePicker;

const HomePage: React.FC = () => {
  const [hashrate, setHashrate] = useState<boolean>(true);
  const [usd, setUsd] = useState<boolean>(false);
  const [startDate, setStartDate] = useState<moment.Moment>(moment('2021-02-15'));
  const [endDate, setEndDate] = useState<moment.Moment>(moment('2021-03-06'));
  const [selectedData, setSelectedData] = useState<IChartData[]>([]);

  const loadData = useCallback(async () => {
    const hashrates = await getHashRates();
    const usds = await getUsds();
    const formattedHashrates: IChartData[] = hashrates.map(hr => { return { date: moment(hr.date).format('YYYY-MM-DD'), hashrate: hr.value }; });
    const formattedUsds: IChartData[] = usds.map(u => { return { date: moment(u.date).format('YYYY-MM-DD'), usd: u.value } });
    let merged: IChartData[] = [];

    for(let i = 0; i < formattedHashrates.length; i++) {
      merged.push({
        ...formattedHashrates[i], 
        ...(formattedUsds.find((itmInner) => itmInner.date === formattedHashrates[i].date))}
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
      <h1>Bitcoin Trend</h1>
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
          <Checkbox onChange={onItemChecked} name="usd" checked={usd}>USD</Checkbox>
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
          {usd && <Line type="monotone" dataKey="usd" stroke="blue" />}
        </LineChart>
      </div>
      <Space size={24}>
        <div>Keyword count: TODO</div>
        <div>TODO B</div>
        <div>TODO C</div>
      </Space>
      <Divider/>
    </div>
  );

  function onItemChecked(e: any) {
    if (e.target.name === 'hashrate') {
      setHashrate(!hashrate);
    } else if (e.target.name === 'usd') {
      setUsd(!usd);
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

export default HomePage;
