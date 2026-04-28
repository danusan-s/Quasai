#include "quasai/core/shape.hpp"
#include <gtest/gtest.h>

using namespace quasai::core;

TEST(ShapeTest, ConstructorInitializerList) {
  Shape shape{2, 3, 4};

  EXPECT_EQ(shape.dimensions(), 3);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);
}

TEST(ShapeTest, ConstructorArray) {
  std::size_t dims[] = {5, 6};
  Shape shape(dims, 2);

  EXPECT_EQ(shape.dimensions(), 2);
  EXPECT_EQ(shape[0], 5);
  EXPECT_EQ(shape[1], 6);
}

TEST(ShapeTest, TotalSize) {
  Shape shape{2, 3, 4};
  EXPECT_EQ(total_size(shape), 24);
}

TEST(ShapeTest, GetStrides) {
  Shape shape{2, 3, 4};
  Strides strides = get_strides(shape);

  EXPECT_EQ(strides.dimensions(), 3);
  EXPECT_EQ(strides[0], 12); // 3*4
  EXPECT_EQ(strides[1], 4);  // 4
  EXPECT_EQ(strides[2], 1);  // 1
}

TEST(ShapeTest, ZeroDimensional) {
  Shape shape{};
  EXPECT_EQ(shape.dimensions(), 0);
  EXPECT_EQ(total_size(shape), 1);
}

TEST(ShapeTest, ExceedMaxDimensions) {
  std::size_t dims[MAX_DIMENSIONS + 1] = {0};
  EXPECT_THROW(Shape shape(dims, MAX_DIMENSIONS + 1),
               std::runtime_error);
}

TEST(ShapeTest, OutOfRangeAccess) {
  Shape shape{2, 3};
  EXPECT_THROW(shape[2], std::out_of_range);
  EXPECT_THROW(shape[100], std::out_of_range);
}
