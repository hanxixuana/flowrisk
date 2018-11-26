# Order Flow Risk Measures

Currently, the packages only has VPIN.

## Installation
The default way is to open a console and execute

    pip install flowrisk

One may also download from here and manually install

    git clone https://github.com/hanxixuana/flowrisk
    cd flowrisk
    python setup.py install


## VPIN
To implement VPIN, we made

    1.  an EWMA estimator of volatility (RecursiveEWMAVol)
    2.  a numpy.ndarray based buckets with bulk classification of volumes in the MA style (RecursiveBulkClassMABuckets)
    3.  a numpy.ndarray based buckets with bulk classification of volumes in the EWMA style 
        (RecursiveBulkClassEWMABuckets)
    4.  a recursive VPIN estimator (RecursiveVPIN)
    5.  a recursive VPIN estimator with VPIN confidence intervals (RecursiveConfVPIN)
    6.  a recursive model using an EWMA estimator of means and RecursiveEWMAVol, for modeling and log 
        innovations of VPINs and for calculating VPINs' confidence intervals (RecursiveEWMABand)
    7.  a one-shoot VPIN estimator for a series of prices (BulkVPIN)
    8.  a one-shoot VPIN estimator for a series of prices with VPIN confidence intervals (BulkConfVPIN)
    9.  various configuration classes (RecursiveVPINConfig, RecursiveConfVPINConfig, BulkVPINConfig, 
        BulkConfVPINConfig)
    
For illustration, we also put the 1-min data of five small caps (CBIO, FBNC, GNC NDLS, QES) and five large caps 
(V, AAPL, NVDA, GS, INTC) from the US stock market. The data covers Nov 12 to Nov 21, 2018. The data can used by, 
for example,

    import flowrisk as fr

    class Config(fr.BulkConfVPINConfig):    
        N_TIME_BAR_FOR_INITIALIZATION = 50
    
    config = Config()
    
    example = fr.examples.USStocks(config)
    symbols = example.list_symbols('small')
    result = example.estimate_vpin_and_conf_interval(symbols[0])
    
    example.draw_price_vpins_and_conf_intervals()

The piece of the code will automatically calculate VPINs and associated confidence intervals of GNC. We also put
prices and volumes together with them into a nice picture, which is saved to ./pics/gnc.png by default. Note that
the calculation of VPINs is fast, but making nice pictures is slow. One may also find out more in test.py.

Note that there are several differences between this implementation and the original paper:

    Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. 
    The Review of Financial Studies, 25(5), 1457-1493.

For example,

    1.  we use an EWMA estimator for the volatility of PnLs, instead of using all samples for estimating the PnL 
        volatility; and
    2.  VPINs are calculated from the very beginning, instead of after a certain number of buckets have been filled.

We made the differences because the core of our package is a recursive estimator of VPIN.  