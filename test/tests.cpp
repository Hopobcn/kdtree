#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <ctime>

#include <boost/random.hpp> // mt19937
#include <boost/nondet_random.hpp> // random_device
#include <boost/random/uniform_real_distribution.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include <kdtree.h>

using namespace spatial_index;

namespace bg = boost::geometry;
namespace bgm = bg::model;
namespace bacc = boost::accumulators;

using TimeAccumulator = bacc::accumulator_set< std::size_t,
                                               bacc::stats< bacc::tag::mean,
                                                            bacc::tag::min,
                                                            bacc::tag::max,
                                                            bacc::tag::variance,
                                                            bacc::tag::median,
                                                            bacc::tag::sum,
                                                            bacc::tag::count>>;

std::ostream& operator<<(std::ostream &out, const TimeAccumulator &acc) {
    out << "count: "    << bacc::count(acc) << "\n";
    out << "min: "      << bacc::min(acc) << "\n";
    out << "max: "      << bacc::max(acc)  << "\n";
    out << "mean: "     << bacc::mean(acc) << "\n";
    out << "variance: " << bacc::variance(acc) << "\n";
    out << "median: "   << bacc::median(acc) << "\n";
    out << "sum: "      << bacc::sum(acc) << "\n";
    return out;
}

template <typename Point>
void randomPoints(std::size_t nr, std::vector<Point>& points) {
    points.resize(nr);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(-10, 10);
    std::uniform_real_distribution<> dis_y(-10, 10);
    for (size_t i = 0; i < nr; i++) {
        Point p = {dis_x(gen), dis_y(gen)};
        points.push_back(p);
    }
}

template <template <typename> class P = std::less >
struct ComparePairFirst {
    template<class T1, class T2> bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right) {
        return P<T1>()(left.first, right.first);
    }
};

template <typename Point>
void LinearSearch(const Point& query, const std::vector<Point>& locations, std::size_t k, std::vector<const Point*>& result) {
    std::vector<std::pair<double, std::size_t>> tmp;
    for (std::size_t i = 0; i < locations.size(); i++) {
        double d = boost::geometry::distance(query, locations.at(i));
        tmp.push_back(std::pair<double, std::size_t>(d, i));
    }
    std::sort(tmp.begin(), tmp.end(), ComparePairFirst<>());
    for (std::size_t i = 0; i < k; i++) {
        std::size_t id =  tmp.at(i).second;
        result.push_back(&locations.at(id));
    }
}

template <typename Point>
class KdTreeTest : public ::testing::Test {
public:
    KdTreeTest()
        : m_gen(m_rd()),
          m_query_count(1000),
          m_point_count(100000) {}

protected:
    virtual void SetUp()
    {
        randomPoints(m_point_count, m_points);
        for (auto&& point : m_points) {
            m_tree.add(&point, &point); // No insert, just adding
        }
        m_tree.build(); // Bulk build
    }

    virtual void TearDown()
    {
        m_tree.clear();
        m_points.clear();
    }

    Point RandomPoint();

    using Data = Point;

    std::vector<Point> m_points;
    kdtree<Data, Point> m_tree;
    std::random_device m_rd;
    std::mt19937 m_gen;
    //std::uniform_real_distribution<> m_dis[Dimensions];
    std::size_t m_query_count;
    std::size_t m_point_count;
};


template <>
class KdTreeTest<bgm::point<double, 2, boost::geometry::cs::cartesian>> : public ::testing::Test {
    using Point = bgm::point<double, 2, boost::geometry::cs::cartesian>;
public:
    KdTreeTest()
        : m_gen(m_rd()),
          m_dis_x(-10, 10),
          m_dis_y(-10, 10),
          m_query_count(1000),
          m_point_count(100000)
    {}

protected:
    virtual void SetUp() {
        randomPoints(m_point_count, m_points);
        for (auto&& point : m_points) {
            m_tree.add(&point, &point); // No insert, just adding
        }
        m_tree.build(); // Bulk build
    }
    virtual void TearDown() {
        m_tree.clear();
        m_points.clear();
    }

    Point RandomPoint() {
        return {m_dis_x(m_gen), m_dis_y(m_gen)};
    }

    using Data = Point;

    std::vector<Point> m_points;
    kdtree<Data, Point> m_tree;
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_real_distribution<> m_dis_x;
    std::uniform_real_distribution<> m_dis_y;
    std::size_t m_query_count;
    std::size_t m_point_count;
};

template <>
class KdTreeTest<bgm::point<double, 3, boost::geometry::cs::cartesian>> : public ::testing::Test {
    using Point = bgm::point<double, 3, boost::geometry::cs::cartesian>;
public:
    KdTreeTest()
        : m_gen(m_rd()),
          m_dis_x(-10, 10),
          m_dis_y(-10, 10),
          m_dis_z(-10, 10),
          m_query_count(1000),
          m_point_count(100000) {}

protected:
    virtual void SetUp()
    {
        randomPoints(m_point_count, m_points);
        for (auto&& point : m_points) {
            m_tree.add(&point, &point); // No insert, just adding
        }
        m_tree.build(); // Bulk build
    }

    virtual void TearDown()
    {
        m_tree.clear();
        m_points.clear();
    }

    Point RandomPoint()
    {
        return {m_dis_x(m_gen), m_dis_y(m_gen), m_dis_z(m_gen)};
    }

    using Data = Point;

    std::vector<Point> m_points;
    kdtree<Data, Point> m_tree;
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::uniform_real_distribution<> m_dis_x;
    std::uniform_real_distribution<> m_dis_y;
    std::uniform_real_distribution<> m_dis_z;
    std::size_t m_query_count;
    std::size_t m_point_count;
};

using PointTypes = ::testing::Types<bgm::point<double, 2, boost::geometry::cs::cartesian>, bgm::point<double, 3, boost::geometry::cs::cartesian>>;
TYPED_TEST_CASE(KdTreeTest, PointTypes);

TYPED_TEST(KdTreeTest, build_tree_performance)
{
}

TYPED_TEST(KdTreeTest, identical_iterative_and_recursive_results)
{
    using Point = TypeParam;
    for (size_t i = 0; i < this->m_query_count; i++) {
        const Point query = this->RandomPoint();
        auto recursive_result = this->m_tree.nearest_recursive(query);
        auto iterative_result = this->m_tree.nearest_iterative(query);
        bool identical = bg::equals(*recursive_result, *iterative_result);
        EXPECT_TRUE(recursive_result != nullptr);
        EXPECT_TRUE(identical) << bg::dsv(*recursive_result) << " != "<< bg::dsv(*iterative_result);
    }
}

TYPED_TEST(KdTreeTest, recursive_performance)
{
    using Point = TypeParam;
    TimeAccumulator time_acc;
    for (std::size_t i = 0; i < this->m_query_count; i++) {
        const Point query = this->RandomPoint();
        auto startTime = std::chrono::high_resolution_clock::now();
        const Point* nearest = this->m_tree.nearest_recursive(query);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        time_acc(t);
    }
    std::cout << "Recursive performance (usec):\n" << time_acc << std::endl;
}

TYPED_TEST(KdTreeTest, iterative_performance)
{
    using Point = TypeParam;
    TimeAccumulator time_acc;
    for (std::size_t i = 0; i < this->m_query_count; i++) {
        const Point query = this->RandomPoint();
        auto startTime = std::chrono::high_resolution_clock::now();
        const Point* nearest = this->m_tree.nearest_iterative(query);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        time_acc(t);
    }
    std::cout << "Iterative performance (usec):\n" << time_acc << std::endl;
}

TYPED_TEST(KdTreeTest, knearest_performance)
{
    using Point = TypeParam;
    TimeAccumulator time_acc;
    std::size_t k = 10;
    std::vector<const Point*> knearest_results;
    for (std::size_t i = 0; i < this->m_query_count; i++) {
        const Point query = this->RandomPoint();
        knearest_results.clear();
        auto startTime = std::chrono::high_resolution_clock::now();
        this->m_tree.knearest(query, k, knearest_results);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        time_acc(t);
    }
    std::cout << "Recursive knearest (" << k << ") performance (usec):\n" << time_acc << std::endl;
}

TYPED_TEST(KdTreeTest, check_knearest_results)
{
    using Point = TypeParam;
    std::vector<const Point*> knearest_results, linear_results;
    std::size_t k = 10;
    for (std::size_t i = 0; i < 10; i++) {
        Point query = this->RandomPoint();
        linear_results.clear();
        knearest_results.clear();
        LinearSearch(query, this->m_points, k, linear_results);
        this->m_tree.knearest(query, k, knearest_results);
        ASSERT_EQ(linear_results.size(), knearest_results.size());
        for (std::size_t j = 0; j < knearest_results.size(); j++) {
            const Point* a = linear_results.at(j);
            const Point* b = knearest_results.at(j);
            bool identical = bg::equals(*a, *b);
            EXPECT_TRUE(identical) << bg::dsv(*a) << " != " << bg::dsv(*b);
        }
    }
}

TEST(WikipediaExample, test) {
    using Point = bgm::d2::point_xy<double>;

    std::vector<Point> points = {{2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7,2}};
    kdtree<Point> tree;
    for (std::size_t i = 0; i < points.size(); i++) {
        tree.add(&points[i], &points[i]);
    }
    tree.build();
    const Point query(5, 6);
    const Point* nearest = tree.nearest_recursive(query);
    const Point expected_result(4, 7);
    bool identical = bg::equals(*nearest, expected_result);
    EXPECT_TRUE(nearest != nullptr);
    EXPECT_TRUE(identical) << bg::dsv(*nearest) << " != " << bg::dsv(expected_result);
}

TEST(DimensionRecursion, subtract) {
    using Point = bgm::d2::point_xy<double>;

    Point p1(2,3);
    Point p2(5,1);
    EXPECT_EQ(-3, util::subtract(p1, p2, 0));
    EXPECT_EQ(2, util::subtract(p1, p2, 1));
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
