import React, { useState, useEffect, useCallback, PureComponent } from 'react';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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

interface ITwoLineData {
  date: string;
  hashrate?: Number;
  usd?: Number;
};

const { RangePicker } = DatePicker;

const HomePage: React.FC = () => {
  const [startDate, setStartDate] = useState<moment.Moment>(moment('2020-01-01'));
  const [endDate, setEndDate] = useState<moment.Moment>(moment('2021-01-15'));
  const [selectedData, setSelectedData] = useState<ITwoLineData[]>([]);

  // const [twoline, setTwoline] = useState<boolean>(true);

  const loadData = useCallback(async () => {
    const twolines = await getHashratepredictions();
    const formattedtwolines: ITwoLineData[] = twolines.map(tl => { return { date: moment(tl.date).format('YYYY-MM-DD'), hashrate: tl.hashrate, usd: tl.prediction }; });
    let merged: ITwoLineData[] = [];

    for(let i = 0; i < formattedtwolines.length; i++) {
      merged.push({
        ...formattedtwolines[i], 
        ...(formattedtwolines.find((itmInner) => itmInner.date === formattedtwolines[i].date))}
      );
    }

    merged = merged.sort(sort);

    let twoLineDataBasedOnDate = merged.filter(d => moment(d.date).isSameOrAfter(startDate) && moment(d.date).isSameOrBefore(endDate));
    setSelectedData(twoLineDataBasedOnDate);
    
  }, [startDate, endDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="container-fluid">
      <h1>Bitcoin TwoLine Chart Trend</h1>
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
        </Space>
      </div>
      <div className="d-inline-block">

        <LineChart width={1200} height={600} data={selectedData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="hashrate" stroke="#8884d8" activeDot={{ r: 8 }} />
          <Line yAxisId="right" type="monotone" dataKey="usd" stroke="#82ca9d" />
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

  // function onItemChecked(e: any) {
  //   if (e.target.name === 'hashrate') {
  //     setHashrate(!hashrate);
  //   } else if (e.target.name === 'hashrateprediction') {
  //     setHashrateprediction(!hashrateprediction);
  //   } else {
  //     return;
  //   }
  // }

  function datesSelected(dates: RangeValue<any>) {
    setStartDate(moment(dates![0]));
    setEndDate(moment(dates![1]));
	}

  function sort(a: ITwoLineData, b: ITwoLineData) {
    return moment(a.date).isAfter(moment(b.date)) ? 1 : -1;
  }

}

export default HomePage;
