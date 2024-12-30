/**
 * Dashboard Analytics - 
 */

'use strict';

(function () {
  let cardColor, headingColor, axisColor, shadeColor, borderColor;

  cardColor = config.colors.white;
  headingColor = config.colors.headingColor;
  axisColor = config.colors.axisColor;
  borderColor = config.colors.borderColor;

  // Total Revenue Report Chart - Line Chart
  // --------------------------------------------------------------------
  const totalRevenueChartEl = document.querySelector('#totalRevenueChart'),
    totalRevenueChartOptions = {
      series: [
        { name: '2021', data: [18, 7, 15, 29, 18, 12, 9] },
        { name: '2020', data: [-13, -18, -9, -14, -5, -17, -15] }
      ],
      chart: {
        height: 350,
        type: 'line', // Changed from bar to line for a modern look
        toolbar: { show: true }
      },
      stroke: {
        curve: 'smooth',
        width: 4,
        colors: ['#FF4560', '#008FFB'] // Updated colors for better contrast
      },
      markers: {
        size: 6,
        colors: ['#FF4560', '#008FFB'],
        strokeColors: '#fff',
        strokeWidth: 2,
        hover: { size: 8 }
      },
      xaxis: {
        categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        labels: { style: { fontSize: '14px', fontWeight: 'bold', colors: axisColor } }
      },
      yaxis: {
        labels: { style: { fontSize: '14px', colors: axisColor } }
      },
      grid: { borderColor: borderColor, strokeDashArray: 4 }
    };
  if (totalRevenueChartEl !== null) {
    const totalRevenueChart = new ApexCharts(totalRevenueChartEl, totalRevenueChartOptions);
    totalRevenueChart.render();
  }

  // Growth Chart - Radial Bar Chart
  // --------------------------------------------------------------------
  const growthChartEl = document.querySelector('#growthChart'),
    growthChartOptions = {
      series: [78],
      labels: ['Growth'],
      chart: { height: 240, type: 'radialBar' },
      plotOptions: {
        radialBar: {
          size: 150,
          startAngle: -120,
          endAngle: 120,
          hollow: { size: '60%' },
          track: { background: cardColor },
          dataLabels: {
            name: { offsetY: 20, color: headingColor, fontSize: '16px', fontWeight: '600' },
            value: { offsetY: -20, color: headingColor, fontSize: '24px', fontWeight: '700' }
          }
        }
      },
      colors: ['#00E396'], // Vibrant green for growth
      stroke: { dashArray: 5 }
    };
  if (growthChartEl !== null) {
    const growthChart = new ApexCharts(growthChartEl, growthChartOptions);
    growthChart.render();
  }

  // Profit Report Line Chart
  // --------------------------------------------------------------------
  const profileReportChartEl = document.querySelector('#profileReportChart'),
    profileReportChartConfig = {
      chart: {
        height: 100,
        type: 'line',
        toolbar: { show: false },
        sparkline: { enabled: true }
      },
      stroke: { width: 4, curve: 'smooth' },
      colors: ['#FEB019'], // Yellow for profit highlights
      series: [{ data: [110, 270, 145, 245, 205, 285] }]
    };
  if (profileReportChartEl !== null) {
    const profileReportChart = new ApexCharts(profileReportChartEl, profileReportChartConfig);
    profileReportChart.render();
  }

  // Order Statistics Chart - Donut
  // --------------------------------------------------------------------
  const chartOrderStatistics = document.querySelector('#orderStatisticsChart'),
    orderChartConfig = {
      chart: { height: 200, type: 'donut' },
      labels: ['Electronics', 'Sports', 'Decor', 'Fashion'],
      series: [85, 15, 50, 50],
      colors: ['#FF4560', '#775DD0', '#00E396', '#FEB019'], // Updated palette
      plotOptions: {
        pie: {
          donut: {
            size: '70%',
            labels: {
              show: true,
              total: {
                show: true,
                label: 'Total',
                fontSize: '16px',
                color: axisColor
              }
            }
          }
        }
      }
    };
  if (chartOrderStatistics !== null) {
    const statisticsChart = new ApexCharts(chartOrderStatistics, orderChartConfig);
    statisticsChart.render();
  }

  // Income Chart - Area Chart
  // --------------------------------------------------------------------
  const incomeChartEl = document.querySelector('#incomeChart'),
    incomeChartConfig = {
      series: [{ data: [24, 21, 30, 22, 42, 26, 35, 29] }],
      chart: {
        height: 250,
        type: 'area',
        toolbar: { show: false }
      },
      stroke: { curve: 'smooth', width: 2 },
      fill: {
        type: 'gradient',
        gradient: { shadeIntensity: 0.7, opacityFrom: 0.5, opacityTo: 0.2 }
      },
      xaxis: {
        categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
        labels: { style: { fontSize: '14px', colors: axisColor } }
      },
      yaxis: {
        labels: { style: { fontSize: '14px', colors: axisColor } }
      },
      grid: { borderColor: borderColor }
    };
  if (incomeChartEl !== null) {
    const incomeChart = new ApexCharts(incomeChartEl, incomeChartConfig);
    incomeChart.render();
  }

  // Weekly Expenses - Radial Chart
  // --------------------------------------------------------------------
  const weeklyExpensesEl = document.querySelector('#expensesOfWeek'),
    weeklyExpensesConfig = {
      series: [65],
      chart: { height: 70, type: 'radialBar' },
      plotOptions: {
        radialBar: {
          startAngle: 0,
          endAngle: 360,
          hollow: { size: '50%' },
          dataLabels: { value: { formatter: val => `$${parseInt(val)}` } }
        }
      },
      colors: ['#FF4560']
    };
  if (weeklyExpensesEl !== null) {
    const weeklyExpenses = new ApexCharts(weeklyExpensesEl, weeklyExpensesConfig);
    weeklyExpenses.render();
  }
})();
