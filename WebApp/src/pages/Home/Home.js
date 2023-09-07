import React from 'react'
import { Link } from 'react-router-dom'

import './Home.css'

export default function Home() {

    return (
        <div className='home-container'>
            <div className='home-header'>
                <h1 className='home-heading'>AI-IS-EveryWhere</h1>
              
            </div>

            <h1 className="description">Tiago's Fitness Buddy</h1>
            <div className="home-main">
                <div className="btn-section">
                    <Link to='/start'>
                        <button
                            className="btn start-btn"
                        >Yoga</button>
                    </Link>
                    <Link to='/tutorials'>
                        <button
                            className="btn start-btn"
                        >Exercise</button>
                    </Link>

                </div>
            </div>
        </div>
    )
}
