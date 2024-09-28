<!--
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask&demo-title=Flask%20%2B%20Vercel&demo-description=Use%20Flask%202%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)

# Flask + Vercel

This example shows how to use Flask 2 on Vercel with Serverless Functions using the [Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python).

## Demo

https://flask-python-template.vercel.app/

## How it Works

This example uses the Web Server Gateway Interface (WSGI) with Flask to enable handling requests on Vercel with Serverless Functions.

## Running Locally

```bash
npm i -g vercel
vercel dev
```

Your Flask application is now available at `http://localhost:3000`.

## One-Click Deploy

Deploy the example using [Vercel](https://vercel.com?utm_source=github&utm_medium=readme&utm_campaign=vercel-examples):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel%2Fexamples%2Ftree%2Fmain%2Fpython%2Fflask&demo-title=Flask%20%2B%20Vercel&demo-description=Use%20Flask%202%20on%20Vercel%20with%20Serverless%20Functions%20using%20the%20Python%20Runtime.&demo-url=https%3A%2F%2Fflask-python-template.vercel.app%2F&demo-image=https://assets.vercel.com/image/upload/v1669994156/random/flask.png)
-->
<img src="https://skillicons.dev/icons?i=python,flask" />
<br>

# A* Algorithm
A Algorithm* is a simple web application built using Flask to demonstrate the A* pathfinding algorithm. The A* algorithm is widely used in computer science, especially in games and robotics, to find the shortest path between two points. This project visualizes how the A* algorithm works in a grid-based environment.

## Table of Contents
1. Overview
2. Features
3. Installation
4. Usage
5. How A* Works
6. Contributing
7. License
   
## Overview
The A Algorithm* project provides a web-based visualization of the A* algorithm, allowing users to set start and end points and visualize the step-by-step process of finding the shortest path on a grid. The application is implemented using Python and Flask for the backend, and JavaScript and HTML for the frontend.

## Features
Interactive Grid: Users can select start and end points on a grid and place obstacles.

Pathfinding Visualization: The app visualizes the step-by-step execution of the A* algorithm.

Dynamic Updates: The grid and path can be dynamically updated as users interact with the interface.

## Installation
To set up the Flask A Algorithm* project on your local machine:

1. Clone the repository: ```git clone https://github.com/joeshwoa/flask-a-star-algorithm.git```
2. Navigate into the project directory: ```cd flask-a-star-algorithm```
3. Create and activate a virtual environment (optional but recommended):
````
python3 -m venv venv
source venv/bin/activate
````
4. Install the required dependencies: ```pip install -r requirements.txt```
5. Run the Flask app: ```flask run```
6. Open the app in your web browser: ```http://localhost:5000```
   
## Usage
Select Start and End Points: Click on the grid to set the starting and ending points.

Place Obstacles: Click on the grid to place obstacles that the algorithm will avoid.

Run the A Algorithm*: Once the start and end points are set, click "Start" to run the A* algorithm and watch the pathfinding process.

Reset: Reset the grid and try different configurations.

## How A* Works
The A* algorithm is a popular pathfinding algorithm that efficiently finds the shortest path between two points on a grid. It combines the best of Dijkstra’s Algorithm and Greedy Best-First Search by using a heuristic to prioritize the most promising nodes. The algorithm evaluates nodes by calculating their cost from the start and estimating the cost to the goal, ensuring that the path found is both optimal and efficient.

### Key concepts:

g(n): The cost to reach the current node from the start.

h(n): The estimated cost to reach the goal from the current node (heuristic).

f(n) = g(n) + h(n): The total cost function used by the A* algorithm to prioritize nodes.

## Contributing
Contributions are welcome! To contribute to the project:

1. Fork the repository.
2. Create a new branch: ```git checkout -b feature-branch```.
3. Make your changes and commit them: ```git commit -m 'Add some feature'```.
4. Push to the branch: ```git push origin feature-branch```.
5. Submit a pull request.
   
## License
This project is licensed under the MIT License. See the LICENSE file for more details.
