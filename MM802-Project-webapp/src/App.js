import React from "react";
import Chart from "./Chart";
class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
    };
  }
  componentDidMount() {
    let startDate = "2013-09-01";
    let endDate = "2021-01-05";
    let api = `https://api.coindesk.com/v1/bpi/historical/close.json?start=${startDate}&end=${endDate}`;

    fetch(api)
      .then((response) => response.json())
      .then((data) => {
        this.setState({
          data: Object.keys(data.bpi).map((date) => {
            return {
              date: new Date(date),
              price: data.bpi[date],
            };
          }),
        });
      })
      .catch((error) => console.log(error));
  }
  render() {
    const { data } = this.state;
    return (
      <div>
        <h2 style={{ textAlign: "center" }}>Bitcoin Price Demo</h2>
        <Chart data={data} />
      </div>
    );
  }
}
export default App;
