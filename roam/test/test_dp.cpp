#include <gtest/gtest.h>
#include "omp.h"

#include "RRTest.h"
#include "DPTest_Basic.h"
#include "DPTest.h"
#include "NodeEnergyTest.h"
#include "CCTest.h"
#include "WarperTest.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
